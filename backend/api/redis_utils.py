# redis_utils.py
import redis
import pandas as pd
import json
import logging
from typing import Optional, List
from config import RedisConfig, get_production_config

logger = logging.getLogger(__name__)

class RedisDataManager:
    def __init__(self, config: Optional[dict] = None, environment: str = "production"):
        # Use provided config or get from environment variables
        if config is None:
            config = RedisConfig.get_redis_config(environment)
        
        if not RedisConfig.validate_config(config):
            raise ValueError("Invalid Redis configuration")
            
        self.redis_config = config
        self._connection = None
        logger.info(f"RedisDataManager initialized for {config['host']}:{config['port']}")
    
    @property
    def connection(self):
        if self._connection is None:
            try:
                self._connection = redis.Redis(**self.redis_config)
                self._connection.ping()
                logger.info("Redis connection established successfully")
            except redis.ConnectionError as e:
                logger.error(f"Redis connection failed: {str(e)}")
                raise
            except redis.AuthenticationError as e:
                logger.error(f"Redis authentication failed: {str(e)}")
                raise
        return self._connection
    
    # ... rest of your methods remain the same ...
    def get_latest_key(self, base_key: str) -> Optional[str]:
        """Get the latest date-based key for a base key"""
        try:
            latest_pointer = self.connection.get(f'{base_key}_latest')
            if latest_pointer:
                return latest_pointer.decode('utf-8')
            
            # Fallback: find most recent date-based key
            pattern = f"{base_key}_*"
            keys = self.connection.keys(pattern)
            if keys:
                dated_keys = [key.decode('utf-8') for key in keys]
                dated_keys.sort(reverse=True)
                return dated_keys[0]
            return None
        except Exception as e:
            logger.error(f"Error finding latest key for {base_key}: {str(e)}")
            return None
    
    def get_dataframe(self, base_key: str, specific_date: str = None) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from Redis using date-based key"""
        try:
            if specific_date:
                redis_key = f"{base_key}_{specific_date}"
            else:
                redis_key = self.get_latest_key(base_key)
            
            if not redis_key:
                logger.warning(f"No Redis key found for {base_key}")
                return None
            
            cached_data = self.connection.get(redis_key)
            if cached_data is None:
                logger.warning(f"No data found in Redis for key: {redis_key}")
                return None
            
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            
            lines = cached_data.strip().split('\n')
            records = [json.loads(line) for line in lines if line.strip()]
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully loaded DataFrame from Redis key: {redis_key}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from Redis key {base_key}: {str(e)}")
            return None
    
    def get_available_dates(self, base_key: str) -> List[str]:
        """Get list of available dates for a base key"""
        try:
            pattern = f"{base_key}_*"
            keys = self.connection.keys(pattern)
            # Filter out the _latest pointer key
            dated_keys = [key.decode('utf-8') for key in keys if not key.decode('utf-8').endswith('_latest')]
            # Extract dates from keys
            dates = [key.replace(f"{base_key}_", "") for key in dated_keys]
            return sorted(dates, reverse=True)  # Most recent first
        except Exception as e:
            logger.error(f"Error getting available dates for {base_key}: {str(e)}")
            return []
    
    def cache_dataframe(self, base_key: str, df: pd.DataFrame, expiration: int = 86400):
        """Cache DataFrame in Redis with date-based key"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            dated_key = f"{base_key}_{current_date}"
            
            # Use your original format: orient='records', lines=True
            df_json = df.to_json(orient='records', lines=True)
            
            # Store the dated data
            self.connection.setex(dated_key, expiration, df_json)
            
            # Update latest pointer
            self.connection.setex(f'{base_key}_latest', expiration, dated_key)
            
            logger.info(f"DataFrame cached in Redis with key: {dated_key}, shape: {df.shape}")
            return dated_key
            
        except Exception as e:
            logger.error(f"Error caching DataFrame in Redis: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.connection.ping()
        except:
            return False