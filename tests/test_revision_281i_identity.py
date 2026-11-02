"""Offline identity and environment safety tests (281i)."""
import pytest

from JBGIdentity import (
    AuthenticatedPrincipal, AuthenticationMode, EntraIdentityProvider,
    IdentitySettings, LocalIdentityProvider,
)


def local_settings(host='127.0.0.1'):
    return IdentitySettings(AuthenticationMode.LOCAL, 'development', host)


def entra_settings(env='production'):
    return IdentitySettings(AuthenticationMode.ENTRA, env, '0.0.0.0')


def test_offline_local_principal_is_stable_and_email_independent():
    local = LocalIdentityProvider(local_settings(), 'developer', email='old@example.org')
    principal = local.current_principal()
    assert principal.principal_key == 'local:9:developer'
    assert principal.principal_key == AuthenticatedPrincipal(
        AuthenticationMode.LOCAL, 'developer', email='new@example.org').principal_key


def test_ipv6_local_loopback_is_accepted():
    assert LocalIdentityProvider(local_settings('::1'), 'developer').current_principal().subject == 'developer'


@pytest.mark.parametrize('environment,host', [
    ('production', '127.0.0.1'), ('development', '0.0.0.0'),
    ('development', 'localhost'), ('development', '192.168.1.8'),
    ('development', '::'),
])
def test_local_auth_cannot_be_enabled_in_unsafe_bindings(environment, host):
    with pytest.raises(ValueError):
        IdentitySettings(AuthenticationMode.LOCAL, environment, host)


def test_missing_or_invalid_mode_fails_closed():
    with pytest.raises(ValueError):
        IdentitySettings(None, 'development', '127.0.0.1')
    with pytest.raises(ValueError):
        LocalIdentityProvider(entra_settings(), 'developer')
    with pytest.raises(ValueError):
        EntraIdentityProvider(local_settings())


def test_entra_maps_verified_claims_without_network_and_uses_tid_oid():
    provider = EntraIdentityProvider(entra_settings())
    claims = {'tid': 'tenant', 'oid': 'person', 'email': 'someone@example.org'}
    principal = provider.from_verified_claims(claims)
    assert principal.principal_key == 'entra:6:tenant:6:person'
    assert principal.email == 'someone@example.org'
    assert provider.from_verified_claims({**claims, 'email': 'changed@example.org'}).principal_key == principal.principal_key
    assert principal.principal_key != LocalIdentityProvider(local_settings(), 'person').current_principal().principal_key


def test_entra_missing_stable_identifier_fails_closed():
    provider = EntraIdentityProvider(entra_settings())
    for claims in ({'tid': 'tenant', 'email': 'person@example.org'}, {'oid': 'person'}, {'tid': '', 'oid': 'person'}):
        with pytest.raises((ValueError, TypeError)):
            provider.from_verified_claims(claims)


def test_key_cannot_collide_on_separator_boundaries():
    a = AuthenticatedPrincipal(AuthenticationMode.ENTRA, 'c', tenant_id='a:b')
    b = AuthenticatedPrincipal(AuthenticationMode.ENTRA, 'b:c', tenant_id='a')
    assert a.principal_key != b.principal_key


def test_local_provider_does_not_accept_tenant_id():
    with pytest.raises(ValueError):
        AuthenticatedPrincipal(AuthenticationMode.LOCAL, 'user', tenant_id='tenant')
