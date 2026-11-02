"""Transport-neutral identity boundary for offline development and Entra (281i).

Authentication happens *before* this module: the Entra adapter accepts only
claims validated by a trusted token-verification component. Never pass request
JSON or unverified JWT claims to it. Scope authorization is a separate concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ipaddress
from typing import Mapping, Any


class AuthenticationMode(str, Enum):
    LOCAL = "local"
    ENTRA = "entra"


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    provider: AuthenticationMode
    subject: str
    tenant_id: str | None = None
    email: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, AuthenticationMode):
            raise TypeError("invalid identity provider")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject required")
        if self.provider is AuthenticationMode.ENTRA:
            if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
                raise ValueError("Entra tenant required")
        elif self.tenant_id is not None:
            raise ValueError("local identities cannot have Entra tenants")
        if self.email is not None and not isinstance(self.email, str):
            raise TypeError("email must be text or None")

    @property
    def principal_key(self) -> str:
        # Length-prefixed segments avoid collisions even if subject contains ':'.
        if self.provider is AuthenticationMode.LOCAL:
            return f"local:{len(self.subject)}:{self.subject}"
        assert self.tenant_id is not None
        return f"entra:{len(self.tenant_id)}:{self.tenant_id}:{len(self.subject)}:{self.subject}"


@dataclass(frozen=True)
class IdentitySettings:
    mode: AuthenticationMode
    environment: str
    bind_host: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AuthenticationMode):
            raise ValueError("explicit identity mode required")
        if self.environment not in ("development", "production"):
            raise ValueError("explicit environment required")
        if not isinstance(self.bind_host, str):
            raise ValueError("bind_host must be a numeric loopback address or network address")
        if self.mode is AuthenticationMode.LOCAL:
            if self.environment != "development":
                raise ValueError("local authentication is forbidden outside development")
            try:
                loopback = ipaddress.ip_address(self.bind_host).is_loopback
            except ValueError:
                loopback = False
            if not loopback:
                raise ValueError("local authentication requires a loopback bind address")


class LocalIdentityProvider:
    """Trusted application bootstrap, not an unauthenticated HTTP login endpoint."""

    def __init__(self, settings: IdentitySettings, subject: str, *, email: str | None = None):
        if not isinstance(settings, IdentitySettings) or settings.mode is not AuthenticationMode.LOCAL:
            raise ValueError("local provider requires explicit local mode")
        self._principal = AuthenticatedPrincipal(AuthenticationMode.LOCAL, subject, email=email)

    def current_principal(self) -> AuthenticatedPrincipal:
        return self._principal


class EntraIdentityProvider:
    """Map **verified** Entra access-token claims; verification is external.

    The caller MUST validate signature, issuer, audience, expiry, authorized
    tenant and relevant scopes/roles before invoking from_verified_claims.
    """

    def __init__(self, settings: IdentitySettings):
        if not isinstance(settings, IdentitySettings) or settings.mode is not AuthenticationMode.ENTRA:
            raise ValueError("Entra provider requires explicit Entra mode")

    def from_verified_claims(self, claims: Mapping[str, Any]) -> AuthenticatedPrincipal:
        if not isinstance(claims, Mapping):
            raise TypeError("verified claims must be a mapping")
        tid, oid = claims.get("tid"), claims.get("oid")
        email = claims.get("email")
        if email is None:
            email = claims.get("preferred_username")
        return AuthenticatedPrincipal(AuthenticationMode.ENTRA, oid, tenant_id=tid, email=email)
