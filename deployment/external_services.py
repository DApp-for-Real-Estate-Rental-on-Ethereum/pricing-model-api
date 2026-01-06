
import httpx
import logging
from typing import Dict, Optional, Any
from deployment.config import settings

logger = logging.getLogger(__name__)

# Timeout for external service calls
TIMEOUT_SEC = 5.0

class ExternalServices:
    """
    Client for communicating with other microservices.
    Uses httpx for async IO.
    """
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Fetch user statistics from user-service."""
        url = f"{settings.USER_SERVICE_URL}/api/v1/users/{user_id}/stats"
        return await self._fetch(url, "user-service", {"rating": 0.0, "score": 100, "verified": False})

    async def get_booking_stats(self, user_id: int) -> Dict[str, Any]:
        """Fetch booking statistics from booking-service."""
        url = f"{settings.BOOKING_SERVICE_URL}/api/bookings/statistics?userId={user_id}"
        return await self._fetch(url, "booking-service", {
            "total": 0, "completed": 0, "cancelled": 0, 
            "avgPrice": 0.0, "avgStayDays": 0.0, "recentLast6Months": 0
        })

    async def get_reclamation_stats(self, user_id: int) -> Dict[str, Any]:
        """Fetch reclamation statistics from reclamation-service."""
        url = f"{settings.RECLAMATION_SERVICE_URL}/api/reclamations/stats?userId={user_id}"
        return await self._fetch(url, "reclamation-service", {
            "totalReceived": 0, "receivedLowSeverity": 0, "receivedMediumSeverity": 0, 
            "receivedHighSeverity": 0, "receivedCriticalSeverity": 0, "receivedOpen": 0,
            "receivedResolvedAgainst": 0, "totalPenaltyPoints": 0, "totalRefundAmount": 0.0
        })

    async def get_payment_stats(self, user_id: int) -> Dict[str, Any]:
        """Fetch payment statistics from payment-service."""
        url = f"{settings.PAYMENT_SERVICE_URL}/api/payments/stats?userId={user_id}"
        return await self._fetch(url, "payment-service", {
            "totalTransactions": 0, "successfulTransactions": 0, 
            "failedTransactions": 0, "avgTransactionAmount": 0.0
        })

    async def _fetch(self, url: str, service_name: str, default: Any) -> Any:
        try:
            # We need to pass headers. Using a dummy ADMIN token or service token would be best.
            # For now, using X-User-Roles: ADMIN as per existing auth logic in services.
            # IN PRODUCTION: Use a secure service-to-service JWT.
            headers = {"X-User-Id": "1", "X-User-Roles": "ADMIN"} 
            
            async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"{service_name} returned 404 for {url}. Using default.")
                    return default
                else:
                    logger.error(f"{service_name} error {response.status_code}: {response.text}")
                    return default
                    
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to {service_name} at {url}: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected error calling {service_name}: {e}")
            return default

external_services = ExternalServices()
