"""
config_loader.py - Carga de configuración desde YAML con interpolación de entorno.
Soporta la sintaxis ${VAR_NAME} para inyectar variables de entorno en el YAML.
"""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Lee el YAML, resuelve ${VAR_NAME} con variables de entorno y retorna el dict.
    Falla explícitamente si alguna variable referenciada no está definida.
    """
    rutaConfig = Path(config_path)
    rutaEnv    = rutaConfig.parent / ".env"
    if rutaEnv.exists():
        load_dotenv(rutaEnv)

    with open(rutaConfig, 'r') as f:
        textoConfig = f.read()

    def _resolverVar(match):
        nombre = match.group(1)
        valor  = os.environ.get(nombre)
        if valor is None:
            raise ValueError(f"Variable de entorno no definida: {nombre}")
        return valor

    textoConfig = re.compile(r'\$\{([^}]+)\}').sub(_resolverVar, textoConfig)
    return yaml.safe_load(textoConfig)


if __name__ == "__main__":
    config = load_config()
    print("Configuración cargada")
    print(f"  host : {config['database']['host']}")
    print(f"  user : {config['database']['user']}")
    print(f"  pass : {'*' * len(config['database']['password'])}")