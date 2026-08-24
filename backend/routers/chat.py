"""AI 追问路由"""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from logging_config import get_logger
from pipeline.sanitizer import sanitize_report

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def create_chat_router(deepseek_agent, knowledge_db=None):
    """工厂函数"""

    @router.post("/ask")
    async def ask_question(data: dict):
        task_id = data.get("task_id", "")
        question = data.get("question", "")
        context = data.get("context", {})

        if not question:
            return {"answer": "请说点什么吧！"}

        task = router.tasks.get(task_id) if hasattr(router, 'tasks') else None
        if task and task.get("result"):
            full_context = {**task["result"]}
        else:
            full_context = {}

        # 分析结果里没有 level/instrument 字段，从 task 顶层补上（否则聊天永远按 beginner 处理）
        if task:
            full_context.setdefault("level", task.get("level", "beginner"))
            full_context.setdefault("instrument", task.get("instrument", "guitar"))

        # 前端 chat 请求中的字段（level 等）优先级高于分析结果，支持会话中切换水平
        full_context.update(context)

        selected_text = data.get("selected_text", "")
        if selected_text:
            full_context["_selected_text"] = selected_text

        # 跟踪对话轮次：第一轮标记 is_first_message=True，后续为 False
        conv_key = f"chat_{task_id}"
        if not hasattr(router, '_conversation_rounds'):
            router._conversation_rounds = {}
        # 定期清理：超过500条时，删除已不存在task_id的轮次记录
        if len(router._conversation_rounds) > 500:
            active_keys = {f"chat_{tid}" for tid in router.tasks}
            stale = [k for k in router._conversation_rounds if k not in active_keys]
            for k in stale:
                del router._conversation_rounds[k]
        round_num = router._conversation_rounds.get(conv_key, 0) + 1
        router._conversation_rounds[conv_key] = round_num
        full_context["_conversation_round"] = round_num

        # 对话历史
        if not hasattr(router, '_conversation_history'):
            router._conversation_history = {}
        conv_history = router._conversation_history.get(conv_key, [])

        # RAG 知识检索 — 使用统一入口
        rag_context = ""
        try:
            if knowledge_db:
                rag_context = knowledge_db.build_rag_for_question(question, full_context)
                if rag_context:
                    logger.info("RAG 注入成功")
        except Exception as e:
            logger.warning(f"RAG 失败: {e}")

        answer = deepseek_agent.ask_question(task_id, question, full_context, rag_context=rag_context, conversation_history=conv_history)
        if answer:
            # 存储对话历史
            conv_history.append({"role": "user", "content": question})
            conv_history.append({"role": "assistant", "content": answer})
            router._conversation_history[conv_key] = conv_history[-20:]  # 保留最近10轮
            return {"answer": answer}
        return {"answer": "抱歉，AI老师暂时无法回答，请稍后再试。"}

    @router.post("/ask/stream")
    async def ask_question_stream(data: dict):
        task_id = data.get("task_id", "")
        question = data.get("question", "")
        context = data.get("context", {})

        if not question:
            async def _empty():
                yield f"data: {json.dumps({'done': True, 'answer': '请说点什么吧！'})}\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")

        task = router.tasks.get(task_id) if hasattr(router, 'tasks') else None
        if task and task.get("result"):
            full_context = {**task["result"]}
        else:
            full_context = {}

        # 分析结果里没有 level/instrument 字段，从 task 顶层补上（否则聊天永远按 beginner 处理）
        if task:
            full_context.setdefault("level", task.get("level", "beginner"))
            full_context.setdefault("instrument", task.get("instrument", "guitar"))

        full_context.update(context)

        selected_text = data.get("selected_text", "")
        if selected_text:
            full_context["_selected_text"] = selected_text

        conv_key = f"chat_{task_id}"
        if not hasattr(router, '_conversation_rounds'):
            router._conversation_rounds = {}
        if len(router._conversation_rounds) > 500:
            active_keys = {f"chat_{tid}" for tid in router.tasks}
            stale = [k for k in router._conversation_rounds if k not in active_keys]
            for k in stale:
                del router._conversation_rounds[k]
        round_num = router._conversation_rounds.get(conv_key, 0) + 1
        router._conversation_rounds[conv_key] = round_num
        full_context["_conversation_round"] = round_num

        # 对话历史
        if not hasattr(router, '_conversation_history'):
            router._conversation_history = {}
        conv_history = router._conversation_history.get(conv_key, [])

        rag_context = ""
        try:
            if knowledge_db:
                rag_context = knowledge_db.build_rag_for_question(question, full_context)
        except Exception:
            pass

        async def _stream():
            full_answer = ""
            try:
                for chunk in deepseek_agent.ask_question_stream(task_id, question, full_context, rag_context=rag_context, conversation_history=conv_history):
                    full_answer += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True, 'answer': full_answer})}\n\n"
                # 存储对话历史
                conv_history.append({"role": "user", "content": question})
                conv_history.append({"role": "assistant", "content": full_answer})
                router._conversation_history[conv_key] = conv_history[-20:]
            except Exception:
                yield f"data: {json.dumps({'done': True, 'answer': full_answer or '抱歉，AI老师暂时无法回答。'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @router.post("/report/stream")
    async def generate_report_stream(data: dict):
        """流式生成AI报告，返回SSE流。"""
        task_id = data.get("task_id", "")

        task = router.tasks.get(task_id) if hasattr(router, 'tasks') else None
        if not task:
            async def _err():
                yield f"data: {json.dumps({'done': True, 'error': '任务不存在'})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream")

        report_inputs = task.get("_report_inputs", {})
        if not report_inputs:
            # Fallback: try to regenerate from task result context
            async def _err():
                yield f"data: {json.dumps({'done': True, 'error': '报告输入数据不可用，请等待分析完成'})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream")

        async def _stream():
            full_text = ""
            try:
                generator = deepseek_agent.generate_report(
                    audio_result=report_inputs.get("audio_result", {}),
                    video_data=report_inputs.get("video_data", {}),
                    instrument=report_inputs.get("instrument", "guitar"),
                    level=report_inputs.get("level", "beginner"),
                    title=report_inputs.get("title", ""),
                    mode="freeplay",
                    audio_diagnosis=report_inputs.get("audio_diagnosis"),
                    stream=True,
                )
                for chunk in generator:
                    full_text += chunk
                    # Sanitize each chunk to prevent temporary display of note names
                    safe_chunk = sanitize_report(chunk)
                    yield f"data: {json.dumps({'chunk': safe_chunk})}\n\n"
                # Final sanitize of complete text (catches boundary-spanning patterns)
                full_text = sanitize_report(full_text)
                # Try to parse complete JSON and return structured result
                try:
                    parsed = deepseek_agent._parse_json(full_text)
                    if parsed:
                        yield f"data: {json.dumps({'done': True, 'parsed': parsed})}\n\n"
                    else:
                        yield f"data: {json.dumps({'done': True, 'raw': full_text})}\n\n"
                except Exception:
                    yield f"data: {json.dumps({'done': True, 'raw': full_text})}\n\n"
            except Exception as e:
                logger.error(f"Stream report error: {e}")
                yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return router


router.tasks = {}
