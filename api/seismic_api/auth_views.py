"""Authentication views — register, login, forgot/reset password, OTP."""
from __future__ import annotations

import logging
import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

logger = logging.getLogger(__name__)

# In-memory OTP store (production: use Redis)
_otp_store: dict[str, dict] = {}


class RegisterView(APIView):
    """POST /api/v1/auth/register/ — Create a new user account."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        username = request.data.get("username", "").strip()
        email = request.data.get("email", "").strip()
        password = request.data.get("password", "")
        first_name = request.data.get("first_name", "")
        last_name = request.data.get("last_name", "")

        if not username or not email or not password:
            return Response(
                {"error": "missing_fields", "detail": "username, email, and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "username_taken", "detail": "A user with that username already exists"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if User.objects.filter(email=email).exists():
            return Response(
                {"error": "email_taken", "detail": "A user with that email already exists"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )

        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                },
                "tokens": {
                    "access": str(refresh.access_token),
                    "refresh": str(refresh),
                },
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    """POST /api/v1/auth/login/ — Authenticate and get JWT tokens."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        username = request.data.get("username", "")
        password = request.data.get("password", "")

        if not username or not password:
            return Response(
                {"error": "missing_credentials", "detail": "username and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response(
                {"error": "invalid_credentials", "detail": "Invalid username or password"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if not user.check_password(password):
            return Response(
                {"error": "invalid_credentials", "detail": "Invalid username or password"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                },
                "tokens": {
                    "access": str(refresh.access_token),
                    "refresh": str(refresh),
                },
            },
            status=status.HTTP_200_OK,
        )


class ForgotPasswordView(APIView):
    """POST /api/v1/auth/forgot-password/ — Send password reset email."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        email = request.data.get("email", "").strip()

        if not email:
            return Response(
                {"error": "missing_email", "detail": "email is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Always return success to prevent email enumeration
        try:
            user = User.objects.get(email=email)
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))

            # In production, send email. For dev, log the link.
            reset_link = f"http://localhost:8000/api/v1/auth/reset-password/?uid={uid}&token={token}"
            logger.info("Password reset link for %s: %s", email, reset_link)

            send_mail(
                subject="SeismicNet — Password Reset",
                message=f"Click to reset your password: {reset_link}",
                from_email="noreply@seismicnet.com",
                recipient_list=[email],
                fail_silently=True,
            )
        except User.DoesNotExist:
            pass

        return Response(
            {"detail": "If an account with that email exists, a reset link has been sent"},
            status=status.HTTP_200_OK,
        )


class ResetPasswordView(APIView):
    """POST /api/v1/auth/reset-password/ — Reset password with token."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        uid = request.data.get("uid", "")
        token = request.data.get("token", "")
        new_password = request.data.get("new_password", "")

        if not uid or not token or not new_password:
            return Response(
                {"error": "missing_fields", "detail": "uid, token, and new_password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user_pk = urlsafe_base64_decode(uid).decode()
            user = User.objects.get(pk=user_pk)
        except (TypeError, ValueError, User.DoesNotExist):
            return Response(
                {"error": "invalid_uid", "detail": "Invalid reset link"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not default_token_generator.check_token(user, token):
            return Response(
                {"error": "invalid_token", "detail": "Invalid or expired reset token"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user.set_password(new_password)
        user.save()

        return Response(
            {"detail": "Password has been reset successfully"},
            status=status.HTTP_200_OK,
        )


class RequestOTPView(APIView):
    """POST /api/v1/auth/request-otp/ — Request a one-time password."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        email = request.data.get("email", "").strip()

        if not email:
            return Response(
                {"error": "missing_email", "detail": "email is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        otp = f"{secrets.randbelow(10**6):06d}"
        _otp_store[email] = {
            "otp": otp,
            "verified": False,
        }

        logger.info("OTP for %s: %s", email, otp)

        send_mail(
            subject="SeismicNet — Your OTP Code",
            message=f"Your one-time password is: {otp}",
            from_email="noreply@seismicnet.com",
            recipient_list=[email],
            fail_silently=True,
        )

        return Response(
            {"detail": "OTP sent to your email"},
            status=status.HTTP_200_OK,
        )


class VerifyOTPView(APIView):
    """POST /api/v1/auth/verify-otp/ — Verify an OTP code."""

    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        email = request.data.get("email", "").strip()
        otp = request.data.get("otp", "")

        if not email or not otp:
            return Response(
                {"error": "missing_fields", "detail": "email and otp are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        stored = _otp_store.get(email)
        if not stored or stored["otp"] != otp:
            return Response(
                {"error": "invalid_otp", "detail": "Invalid or expired OTP"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        stored["verified"] = True
        return Response(
            {"detail": "OTP verified successfully"},
            status=status.HTTP_200_OK,
        )


class ProfileView(APIView):
    """GET /api/v1/auth/profile/ — Get current user profile."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        user = request.user
        return Response(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_staff": user.is_staff,
                "date_joined": user.date_joined.isoformat(),
            },
            status=status.HTTP_200_OK,
        )
