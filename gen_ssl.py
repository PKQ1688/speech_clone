from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID
from datetime import datetime, timedelta

# 生成私钥
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# 生成CSR（证书请求）
csr = x509.CertificateSigningRequestBuilder().subject_name(
    x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),  # 这里使用"US"作为美国的2字符代码
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Your State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Your Locality"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Your Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"Your Common Name"),
    ])
).sign(private_key, hashes.SHA256(), default_backend())

# 生成自签名证书
now = datetime.utcnow()
cert = x509.CertificateBuilder().subject_name(
    x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),  # 这里使用"US"作为美国的2字符代码
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Your State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Your Locality"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Your Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"Your Common Name"),
    ])
).issuer_name(
    x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),  # 这里使用"US"作为美国的2字符代码
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Your State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Your Locality"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Your Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"Your Common Name"),
    ])
).public_key(
    csr.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    now
).not_valid_after(
    now + timedelta(days=365)  # 证书有效期为1年
).sign(private_key, hashes.SHA256(), default_backend())

# 将私钥和证书保存到文件
with open("private_key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

with open("certificate.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

