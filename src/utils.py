"""
utils.py
========
Funciones auxiliares y utilidades.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path


def setup_logging(config):
    """
    Configura el sistema de logging.
    
    Args:
        config: Diccionario con configuración
        
    Returns:
        Logger configurado
    """
    log_config = config['logging']
    
    # Crear directorio de logs si no existe
    log_file = None
    if log_config.get('file', True):
        log_dir = Path(log_config.get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Nombre de archivo con fecha
        log_file = log_dir / f"predictions_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Configurar formato
    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Nivel de logging
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Configurar handlers
    handlers = []
    
    if log_config.get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    if log_config.get('file', True) and log_file:
        file_handler = logging.FileHandler(
            log_file,
            encoding='utf-8',
            errors='replace'  # Ignora caracteres que no se puedan codificar
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configurar logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Fuerza reconfiguración si ya existe
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado: nivel={log_config.get('level')}")
    
    return logger


def send_email_alert(config, subject=None, body=None, message=None):
    """
    Envía email de alerta en caso de error.
    
    Args:
        config: Diccionario con configuración
        subject: Asunto del email (opcional)
        body: Cuerpo del email (opcional)
        message: Mensaje de error (compatibilidad con versión anterior)
    """
    notif_config = config.get('notifications', {})
    
    if not notif_config.get('email_on_error', False):
        return
    
    # Si se pasó 'message' por compatibilidad, usarlo como body
    if message and not body:
        body = message
    
    try:
        # Configurar email
        msg = MIMEMultipart()
        msg['From'] = notif_config['smtp_user']
        msg['To'] = notif_config['email_to']
        
        # Subject por defecto
        if not subject:
            subject = f"ALERTA - Sistema Predictivo - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        msg['Subject'] = subject
        
        # Body por defecto
        if not body:
            body = f"""
Ha ocurrido un error en el sistema de mantenimiento predictivo.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Por favor, revisar los logs para más detalles.
            """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Enviar email
        server = smtplib.SMTP(notif_config['smtp_host'], notif_config['smtp_port'])
        server.starttls()
        server.login(notif_config['smtp_user'], notif_config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        logging.info("Email de alerta enviado")
        
    except Exception as e:
        logging.error(f"Error al enviar email: {e}")


def validate_dataframe(df, required_columns):
    """
    Valida que un DataFrame tenga las columnas requeridas.
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas
        
    Returns:
        True si válido, False en caso contrario
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        logging.error(f"Columnas faltantes: {missing_cols}")
        return False
    
    return True