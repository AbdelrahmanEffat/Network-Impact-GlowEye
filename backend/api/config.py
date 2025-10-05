# config.py
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class RedisConfig:
    """Redis configuration using environment variables"""
    
    @staticmethod
    def get_redis_config(environment: str = "production") -> dict:
        """Get Redis configuration for specified environment"""
        prefix = "" if environment == "production" else f"{environment.upper()}_"
        
        config = {
            'host': os.getenv(f'{prefix}REDIS_HOST', 'localhost'),
            'port': int(os.getenv(f'{prefix}REDIS_PORT', 6379)),
            'password': os.getenv(f'{prefix}REDIS_PASSWORD'),
            'db': int(os.getenv(f'{prefix}REDIS_DB', 0)),
            'decode_responses': False,
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }
        
        # Remove password if it's None (allows connection without password)
        if config['password'] is None:
            config.pop('password')
            
        return config
    
    @staticmethod
    def validate_config(config: dict) -> bool:
        """Validate that required configuration is present"""
        required = ['host', 'port']
        return all(key in config and config[key] for key in required)

# Convenience functions
def get_production_config():
    return RedisConfig.get_redis_config("production")

def get_development_config():
    return RedisConfig.get_redis_config("development")