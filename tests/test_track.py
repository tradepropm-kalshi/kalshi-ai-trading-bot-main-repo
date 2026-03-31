import asyncio
import os
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime

from src.jobs.track import run_tracking
from src.utils.database import DatabaseManager, Position
import aiosqlite

TEST_DB = "track_test.db"

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

async def get_position_by_market_id_any_status(db_manager: DatabaseManager, market_id: str):
    """Helper function to get position regardless of status for testing."""
    async with aiosqlite.connect(db_manager.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM positions WHERE market_id = ? LIMIT 1", (market_id,))
        row = await cursor.fetchone()
        if row:
            position_dict = dict(row)
            position_dict['timestamp'] = datetime.fromisoformat(position_dict['timestamp'])
            return Position(**position_dict)
        return None

@patch('src.jobs.track.KalshiClient')
async def test_run_tracking_closes_position(mock_kalshi_client):
    """
    Test that the tracking job correctly identifies a closed market,
    updates the position status, and creates a trade log.
    """
    # Arrange: Setup database with a live position
    db_path = TEST_DB
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db_manager = DatabaseManager(db_path=db_path)
    await db_manager.initialize()

    test_position = Position(
        market_id="TRACK-TEST-1",
        side="YES",
        entry_price=0.40,
        quantity=5,
        timestamp=datetime.now(),
        rationale="A position to be tracked",
        confidence=0.75,
        live=True,
        status="open"
    )
    position_id = await db_manager.add_position(test_position)
    test_position.id = position_id

    # Mock the KalshiClient to return a closed market that resolved to 'YES'
    mock_api = mock_kalshi_client.return_value
    mock_api.get_market = AsyncMock(return_value={
        "market": {
            "status": "closed",
            "result": "YES"
        }
    })
    mock_api.close = AsyncMock()

    try:
        # Act: Run the tracking job
        await run_tracking(db_manager=db_manager)

        # Assert
        # 1. Check if the position is now 'closed'
        updated_position = await get_position_by_market_id_any_status(db_manager, "TRACK-TEST-1")
        assert updated_position is not None, "Position should still exist."
        assert updated_position.status == "closed", "Position should be marked as closed."

        # 2. Check if a trade log was created
        trade_logs = await db_manager.get_all_trade_logs()
        assert len(trade_logs) == 1, "A trade log should have been created."
        
        log = trade_logs[0]
        assert log.market_id == "TRACK-TEST-1"
        assert log.pnl == (1.0 - 0.40) * 5, "PnL should be calculated correctly for a win."

        # 3. Verify mocks - Updated for new sell limit order functionality
        # The tracking system now calls get_market multiple times:
        # 1. Once for profit-taking check
        # 2. Once for stop-loss check  
        # 3. Once for traditional exit strategy check
        assert mock_api.get_market.call_count >= 1, "get_market should be called at least once"
        mock_api.close.assert_called_once()

    finally:
        # Teardown
        if os.path.exists(db_path):
            os.remove(db_path) 