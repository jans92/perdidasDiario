"""
utils.py - Logging, alertas por email y validación de DataFrames.
"""

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def setup_logging(config: dict) -> logging.Logger:
    """
    Configura logging a consola y/o fichero según config['logging'].
    Retorna el logger raíz del módulo.
    """
    logConfig = config['logging']

    handlers = []

    if logConfig.get('console', True):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(logConfig.get(
            'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )))
        handlers.append(ch)

    if logConfig.get('file', True):
        dirLogs = Path(logConfig.get('log_dir', './logs'))
        dirLogs.mkdir(exist_ok=True)
        rutaLog = dirLogs / f"predictions_{datetime.now().strftime('%Y-%m-%d')}.log"
        fh = logging.FileHandler(rutaLog, encoding='utf-8', errors='replace')
        fh.setFormatter(logging.Formatter(logConfig.get(
            'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )))
        handlers.append(fh)

    logging.basicConfig(
        level=getattr(logging, logConfig.get('level', 'INFO')),
        handlers=handlers,
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado: nivel={logConfig.get('level')}")
    return logger


def send_email_alert(config: dict, subject: str = None, body: str = None) -> None:
    """
    Envía un email de alerta si notifications.email_on_error está activo.
    No lanza excepción si el envío falla — solo lo registra en el log.
    """
    notif = config.get('notifications', {})
    if not notif.get('email_on_error', False):
        return

    msg             = MIMEMultipart()
    msg['From']     = notif['smtp_user']
    msg['To']       = notif['email_to']
    msg['Subject']  = subject or f"ALERTA — Sistema Predictivo — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg.attach(MIMEText(
        body or f"Error en el sistema de mantenimiento predictivo.\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Revisar logs para más detalles.",
        'plain'
    ))

    try:
        srv = smtplib.SMTP(notif['smtp_host'], notif['smtp_port'])
        srv.starttls()
        srv.login(notif['smtp_user'], notif['smtp_password'])
        srv.send_message(msg)
        srv.quit()
        logging.info("Email de alerta enviado")
    except Exception as e:
        logging.error(f"Error enviando email de alerta: {e}")


def validarDataFrame(df, columnasRequeridas: list) -> bool:
    """Retorna True si df contiene todas las columnas requeridas, False si falta alguna."""
    faltantes = set(columnasRequeridas) - set(df.columns)
    if faltantes:
        logging.error(f"Columnas faltantes en DataFrame: {faltantes}")
        return False
    return True