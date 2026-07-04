"""Tests for authentication endpoints."""
from __future__ import annotations

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestRegisterEndpoint(TestCase):
    """Tests for POST /api/v1/auth/register/."""

    def test_register_creates_user(self) -> None:
        """Register creates a new user and returns tokens."""
        client = Client()
        response = client.post(
            "/api/v1/auth/register/",
            {"username": "testuser", "email": "test@example.com", "password": "securepass123"},
            content_type="application/json",
        )
        assert response.status_code == 201
        data = response.json()
        assert data["user"]["username"] == "testuser"
        assert "tokens" in data
        assert "access" in data["tokens"]
        assert User.objects.filter(username="testuser").exists()

    def test_register_duplicate_username(self) -> None:
        """Register rejects duplicate username."""
        User.objects.create_user("testuser", "t@example.com", "pass")
        client = Client()
        response = client.post(
            "/api/v1/auth/register/",
            {"username": "testuser", "email": "other@example.com", "password": "pass"},
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.json()["error"] == "username_taken"

    def test_register_missing_fields(self) -> None:
        """Register rejects missing required fields."""
        client = Client()
        response = client.post(
            "/api/v1/auth/register/",
            {"username": "testuser"},
            content_type="application/json",
        )
        assert response.status_code == 400


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestLoginEndpoint(TestCase):
    """Tests for POST /api/v1/auth/login/."""

    def setUp(self) -> None:
        self.user = User.objects.create_user("testuser", "t@example.com", "pass1234")

    def test_login_success(self) -> None:
        """Login returns tokens for valid credentials."""
        client = Client()
        response = client.post(
            "/api/v1/auth/login/",
            {"username": "testuser", "password": "pass1234"},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["username"] == "testuser"
        assert "access" in data["tokens"]

    def test_login_wrong_password(self) -> None:
        """Login rejects wrong password."""
        client = Client()
        response = client.post(
            "/api/v1/auth/login/",
            {"username": "testuser", "password": "wrong"},
            content_type="application/json",
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self) -> None:
        """Login rejects nonexistent user."""
        client = Client()
        response = client.post(
            "/api/v1/auth/login/",
            {"username": "nobody", "password": "pass"},
            content_type="application/json",
        )
        assert response.status_code == 401


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestPasswordResetFlow(TestCase):
    """Tests for forgot-password → reset-password flow."""

    def setUp(self) -> None:
        self.user = User.objects.create_user("testuser", "t@example.com", "oldpass123")

    def test_forgot_password_returns_success(self) -> None:
        """Forgot password always returns success (prevents enumeration)."""
        client = Client()
        response = client.post(
            "/api/v1/auth/forgot-password/",
            {"email": "t@example.com"},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_forgot_password_nonexistent_email(self) -> None:
        """Forgot password returns success even for unknown email."""
        client = Client()
        response = client.post(
            "/api/v1/auth/forgot-password/",
            {"email": "unknown@example.com"},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_reset_password_missing_fields(self) -> None:
        """Reset password rejects missing fields."""
        client = Client()
        response = client.post(
            "/api/v1/auth/reset-password/",
            {"uid": "", "token": "", "new_password": ""},
            content_type="application/json",
        )
        assert response.status_code == 400


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestOTPFlow(TestCase):
    """Tests for request-otp → verify-otp flow."""

    def test_request_otp_returns_success(self) -> None:
        """Request OTP returns success."""
        client = Client()
        response = client.post(
            "/api/v1/auth/request-otp/",
            {"email": "test@example.com"},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_verify_otp_invalid(self) -> None:
        """Verify OTP rejects invalid code."""
        client = Client()
        response = client.post(
            "/api/v1/auth/verify-otp/",
            {"email": "test@example.com", "otp": "000000"},
            content_type="application/json",
        )
        assert response.status_code == 400


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestProfileEndpoint(TestCase):
    """Tests for GET /api/v1/auth/profile/."""

    def test_profile_requires_auth(self) -> None:
        """Profile endpoint requires authentication."""
        client = Client()
        response = client.get("/api/v1/auth/profile/")
        assert response.status_code == 401

    def test_profile_returns_user_data(self) -> None:
        """Profile returns authenticated user data via JWT."""
        user = User.objects.create_user("testuser", "t@example.com", "pass1234")
        client = Client()
        # Login to get a valid JWT token
        login_response = client.post(
            "/api/v1/auth/login/",
            {"username": "testuser", "password": "pass1234"},
            content_type="application/json",
        )
        token = login_response.json()["tokens"]["access"]
        response = client.get(
            "/api/v1/auth/profile/",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "t@example.com"
