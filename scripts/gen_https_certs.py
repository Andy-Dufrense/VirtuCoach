"""生成 VirtuCoach-Graduate 局域网 HTTPS 证书（P0-4，幂等，可重复运行）。

输出: certs/virtucoach-ca.{crt,key} + certs/virtucoach-server.{crt,key}

- CA 只生成一次（手机信任后不要随便换）；服务器证书每次重跑都会按当前
  本机 IP 重新签发，换网络 / 换电脑后重跑一次即可。
- 自动收集本机全部非回环 IPv4 加入 SAN，也可以 --host 手动补充。
- 依赖: 项目便携 Python（同级 Python310）自带 cryptography，无网络依赖。

用法:
    python scripts/gen_https_certs.py
    python scripts/gen_https_certs.py --host 192.168.0.95 --host 10.0.0.8
"""
import argparse
import datetime
import ipaddress
import socket
import sys
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
except ImportError:
    sys.stderr.write("缺少 cryptography，请用项目同级便携 Python 运行本脚本。\n")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent
CERTS_DIR = BASE_DIR / "certs"

CA_KEY = CERTS_DIR / "virtucoach-ca.key"
CA_CRT = CERTS_DIR / "virtucoach-ca.crt"
SERVER_KEY = CERTS_DIR / "virtucoach-server.key"
SERVER_CRT = CERTS_DIR / "virtucoach-server.crt"

CA_CN = "VirtuCoach Local CA"


def collect_ipv4() -> list[str]:
    """收集本机所有非回环 IPv4 地址。"""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not ipaddress.ip_address(addr).is_loopback:
                ips.add(addr)
    except OSError:
        pass
    try:
        _, _, aliases = socket.gethostbyname_ex(socket.gethostname())
        for alias in aliases:
            try:
                ip = socket.gethostbyname(alias)
                if not ipaddress.ip_address(ip).is_loopback:
                    ips.add(ip)
            except OSError:
                pass
    except OSError:
        pass
    return sorted(ips)


def ensure_ca() -> tuple:
    """返回 (ca_cert, ca_key)；已存在则复用，否则新建。"""
    if CA_KEY.exists() and CA_CRT.exists():
        key = serialization.load_pem_private_key(CA_KEY.read_bytes(), password=None)
        crt = x509.load_pem_x509_certificate(CA_CRT.read_bytes())
        return crt, key

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, CA_CN),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VirtuCoach"),
    ])
    crt = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                key_cert_sign=True, crl_sign=True, digital_signature=False,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    CA_KEY.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CA_CRT.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    print(f"[*] 已生成新 CA: {CA_CRT}")
    return crt, key


def issue_server(ca_crt, ca_key, extra_hosts: list[str]) -> None:
    """按当前 IP 签发服务器证书（每次重跑都刷新）。"""
    ips = collect_ipv4()
    for h in extra_hosts:
        try:
            ip = ipaddress.ip_address(h)
            if not ip.is_loopback and str(ip) not in ips:
                ips.append(str(ip))
        except ValueError:
            pass

    san: list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    for ip in sorted(set(ips)):
        san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    for h in extra_hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san.append(x509.DNSName(h))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    cn = extra_hosts[0] if extra_hosts else (ips[0] if ips else "localhost")
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VirtuCoach"),
    ])
    crt = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_crt.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    SERVER_KEY.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    SERVER_CRT.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    san_desc = ", ".join(
        f"IP:{x.value}" if isinstance(x.value, ipaddress.IPv4Address) else f"DNS:{x.value}"
        for x in san
    )
    print(f"[*] 服务器证书已签发: {SERVER_CRT}")
    print(f"[*] SAN: {san_desc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 VirtuCoach 局域网 HTTPS 证书")
    ap.add_argument("--host", action="append", default=[], help="额外加入证书的主机名/IP（可多次传）")
    args = ap.parse_args()

    ca_crt, ca_key = ensure_ca()
    issue_server(ca_crt, ca_key, args.host)

    print()
    print("证书目录:", CERTS_DIR)
    print("手机安装 CA: 浏览器打开 https://<本机IP>:1443/ca.crt 下载并信任 virtucoach-ca.crt")


if __name__ == "__main__":
    main()
