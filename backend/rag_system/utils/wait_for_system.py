# backend/rag_system/utils/wait_for_system.py
import time
import argparse
import logging
import httpx # Using httpx for async requests

# Basic logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_service(url: str, service_name: str) -> bool:
    """Checks if a service is responsive."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0) # 5 second timeout for check
        if 200 <= response.status_code < 300:
            logger.info(f"{service_name} at {url} is responsive (Status: {response.status_code}).")
            return True
        else:
            logger.warning(f"{service_name} at {url} returned status {response.status_code}.")
            return False
    except httpx.RequestError as e:
        logger.warning(f"Failed to connect to {service_name} at {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking {service_name} at {url}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Wait for RAG system components to be ready.")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/health", # Default to health endpoint of the API
        help="URL of the RAG API health endpoint to check."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120, # Total timeout in seconds
        help="Maximum time (in seconds) to wait for services."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5, # Interval between checks in seconds
        help="Time (in seconds) to wait between checks."
    )

    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"Waiting for RAG API at {args.api_url} to be ready (timeout: {args.timeout}s)...")

    while time.time() - start_time < args.timeout:
        if await check_service(args.api_url, "RAG API"):
            logger.info("RAG API is ready. Proceeding with application startup.")
            exit(0) # Success

        logger.info(f"RAG API not ready yet. Retrying in {args.interval} seconds...")
        time.sleep(args.interval) # Synchronous sleep is fine for this script

    logger.error(f"Timeout: RAG API at {args.api_url} did not become ready within {args.timeout} seconds.")
    exit(1) # Failure

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
