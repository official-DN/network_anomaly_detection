"""
MongoDB database connection and collection management using Motor (async driver).
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from backend.config import settings
from backend.utils.logger import logger
from typing import Optional


class Database:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None


db_instance = Database()


async def connect_db() -> None:
    """Connect to MongoDB."""
    try:
        db_instance.client = AsyncIOMotorClient(settings.mongodb_url)
        db_instance.db = db_instance.client[settings.database_name]
        # Verify connection
        await db_instance.client.admin.command("ping")
        logger.info(f"Connected to MongoDB: {settings.database_name}")
        await create_indexes()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.warning("Running in mock-DB mode (no MongoDB available)")


async def disconnect_db() -> None:
    """Disconnect from MongoDB."""
    if db_instance.client:
        db_instance.client.close()
        logger.info("Disconnected from MongoDB")


async def create_indexes() -> None:
    """Create database indexes for performance."""
    if db_instance.db is None:
        return
    try:
        # traffic_events indexes
        await db_instance.db.traffic_events.create_index("timestamp")
        await db_instance.db.traffic_events.create_index("src_ip")
        await db_instance.db.traffic_events.create_index("prediction")

        # alerts indexes
        await db_instance.db.alerts.create_index("timestamp")
        await db_instance.db.alerts.create_index("risk_level")
        await db_instance.db.alerts.create_index("status")

        # model_metrics indexes
        await db_instance.db.model_metrics.create_index("model_name")
        await db_instance.db.model_metrics.create_index("created_at")

        logger.info("Database indexes created")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


def get_db() -> Optional[AsyncIOMotorDatabase]:
    """Get database instance."""
    return db_instance.db
