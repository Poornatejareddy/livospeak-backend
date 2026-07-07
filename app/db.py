import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError
from app.config import MONGODB_URI

logger = logging.getLogger(__name__)

# Global state for database connection
db_client: Optional[AsyncIOMotorClient] = None
db = None
use_in_memory = False
in_memory_history: List[Dict[str, Any]] = []

async def init_db():
    global db_client, db, use_in_memory
    
    if not MONGODB_URI:
        logger.warning("MONGODB_URI is not set. Falling back to In-Memory history storage.")
        use_in_memory = True
        return
        
    try:
        # Create client with a 2-second connection timeout for fast failover/fallback
        db_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        try:
            db = db_client.get_default_database()
        except Exception:
            db = db_client.get_database("livospeak")
        
        # Ping database to verify connection
        await db_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
    except (ServerSelectionTimeoutError, Exception) as e:
        logger.warning(
            f"Failed to connect to MongoDB ({str(e)}). "
            "Falling back to In-Memory history storage for this session."
        )
        use_in_memory = True
        db_client = None
        db = None

def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert MongoDB ObjectId to string for JSON serialization."""
    if not doc:
        return doc
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

async def save_analysis(result: Dict[str, Any]) -> str:
    """
    Saves speech analysis result to history.
    Inserts a timestamp and returns the record ID.
    """
    record = dict(result)
    record["timestamp"] = datetime.utcnow()
    
    if use_in_memory:
        # Generate a mock ObjectId string
        record["id"] = f"mem_{int(datetime.utcnow().timestamp())}_{len(in_memory_history)}"
        in_memory_history.append(record)
        logger.info(f"Saved analysis to in-memory history (ID: {record['id']})")
        return record["id"]
        
    try:
        res = await db.analyses.insert_one(record)
        logger.info(f"Saved analysis to MongoDB (ID: {str(res.inserted_id)})")
        return str(res.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save analysis to MongoDB: {str(e)}. Falling back to in-memory.")
        record["id"] = f"mem_{int(datetime.utcnow().timestamp())}"
        in_memory_history.append(record)
        return record["id"]

async def get_analyses_history(limit: int = 15) -> List[Dict[str, Any]]:
    """
    Retrieves summary items of recent analyses sorted by timestamp descending.
    """
    if use_in_memory:
        # Sort in-memory history by timestamp desc and map to summary
        sorted_mem = sorted(in_memory_history, key=lambda x: x.get("timestamp", datetime.utcnow()), reverse=True)
        return [
            {
                "id": doc.get("id"),
                "timestamp": doc.get("timestamp").isoformat() if isinstance(doc.get("timestamp"), datetime) else doc.get("timestamp"),
                "scores": doc.get("scores"),
                "speech_rate": doc.get("speech_rate"),
                "duration": doc.get("duration"),
                "transcript": doc.get("transcript")[:60] + "..." if len(doc.get("transcript", "")) > 60 else doc.get("transcript")
            }
            for doc in sorted_mem[:limit]
        ]
        
    try:
        cursor = db.analyses.find(
            {}, 
            {
                "scores": 1, 
                "speech_rate": 1, 
                "duration": 1, 
                "transcript": 1, 
                "timestamp": 1
            }
        ).sort("timestamp", -1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        summaries = []
        for d in docs:
            serialized = serialize_doc(d)
            # Shorten transcript for history overview
            tx = serialized.get("transcript", "")
            serialized["transcript"] = tx[:60] + "..." if len(tx) > 60 else tx
            # Convert datetime to string
            if isinstance(serialized.get("timestamp"), datetime):
                serialized["timestamp"] = serialized["timestamp"].isoformat()
            summaries.append(serialized)
        return summaries
    except Exception as e:
        logger.error(f"Failed to fetch analysis history from MongoDB: {str(e)}")
        return []

async def get_analysis_detail(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves full analysis detail by ID.
    """
    if use_in_memory:
        for doc in in_memory_history:
            if doc.get("id") == analysis_id:
                return doc
        return None
        
    try:
        # Support both MongoDB ObjectId and Mock In-Memory ID
        if analysis_id.startswith("mem_"):
            for doc in in_memory_history:
                if doc.get("id") == analysis_id:
                    return doc
            return None
            
        doc = await db.analyses.find_one({"_id": ObjectId(analysis_id)})
        if doc:
            serialized = serialize_doc(doc)
            if isinstance(serialized.get("timestamp"), datetime):
                serialized["timestamp"] = serialized["timestamp"].isoformat()
            return serialized
        return None
    except Exception as e:
        logger.error(f"Failed to fetch analysis detail ({analysis_id}): {str(e)}")
        # Check in memory just in case
        for doc in in_memory_history:
            if doc.get("id") == analysis_id:
                return doc
        return None
