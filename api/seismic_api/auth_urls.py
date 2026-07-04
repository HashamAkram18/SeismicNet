"""URL configuration for authentication endpoints."""
from django.urls import path

from seismic_api.auth_views import (
    ForgotPasswordView,
    LoginView,
    ProfileView,
    RegisterView,
    RequestOTPView,
    ResetPasswordView,
    VerifyOTPView,
)

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("login/", LoginView.as_view(), name="login"),
    path("forgot-password/", ForgotPasswordView.as_view(), name="forgot-password"),
    path("reset-password/", ResetPasswordView.as_view(), name="reset-password"),
    path("request-otp/", RequestOTPView.as_view(), name="request-otp"),
    path("verify-otp/", VerifyOTPView.as_view(), name="verify-otp"),
    path("profile/", ProfileView.as_view(), name="profile"),
]
