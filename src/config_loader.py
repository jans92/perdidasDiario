"""
Cargador de configuración con soporte para variables de entorno
"""
import os
import yaml
import re
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga configuración desde YAML y reemplaza variables de entorno.
    
    Soporta sintaxis ${VAR_NAME} en el YAML.
    """
    # Cargar .env si existe
    env_path = Path(config_path).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Leer YAML
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Reemplazar ${VAR_NAME} con valores de entorno
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replace_env_var(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Variable de entorno no definida: {var_name}")
        return value
    
    config_text = pattern.sub(replace_env_var, config_text)
    
    # Parsear YAML
    config = yaml.safe_load(config_text)
    
    return config


# Uso simple
if __name__ == "__main__":
    config = load_config()
    print(" Configuración cargada")
    print(f"   DB Host: {config['database']['host']}")
    print(f"   DB User: {config['database']['user']}")
    print(f"   DB Pass: {'*' * len(config['database']['password'])}")  # No mostrar