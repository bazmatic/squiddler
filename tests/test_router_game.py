"""The game builds the router from INCHWORM_ROUTER."""
import logging
from types import SimpleNamespace

from creatures.inchworm_neural import driver_for
from learning.adapters import inchworm as adapter
from learning.policy import random_policy
from learning.router import RoutedPolicy


def enc():
    return SimpleNamespace(world=SimpleNamespace())


def saved(tmp_path, seed):
    path = tmp_path/f'policy{seed}.npz'
    random_policy(adapter.SPEC,seed,hidden=8,embed=4).save(path)
    return path


def test_driver_for_uses_the_router_when_the_variable_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv('INCHWORM_ROUTER',f'{saved(tmp_path,1)},{saved(tmp_path,2)}')
    assert isinstance(driver_for(enc()).policy,RoutedPolicy)


def test_driver_for_is_unchanged_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv('INCHWORM_ROUTER',raising=False)
    monkeypatch.setenv('INCHWORM_POLICY',str(saved(tmp_path,1)))
    driver = driver_for(enc())
    assert driver is not None and not isinstance(driver.policy,RoutedPolicy)


def test_a_bad_router_path_falls_back_to_the_single_policy_with_a_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('INCHWORM_ROUTER',f'{tmp_path}/missing.npz,{saved(tmp_path,2)}')
    monkeypatch.setenv('INCHWORM_POLICY',str(saved(tmp_path,1)))
    with caplog.at_level(logging.WARNING):
        driver = driver_for(enc())
    assert driver is not None and not isinstance(driver.policy,RoutedPolicy) and 'router' in caplog.text.lower()


def test_the_shipped_walker_and_leaper_make_the_default_router(monkeypatch):
    monkeypatch.delenv('INCHWORM_ROUTER',raising=False)
    monkeypatch.delenv('INCHWORM_POLICY',raising=False)
    assert isinstance(driver_for(enc()).policy,RoutedPolicy)


def test_an_empty_policy_variable_still_selects_the_scripted_walk(monkeypatch):
    monkeypatch.delenv('INCHWORM_ROUTER',raising=False)
    monkeypatch.setenv('INCHWORM_POLICY','')
    assert driver_for(enc()) is None
