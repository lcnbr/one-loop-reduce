"""Interpreter checks for the shared Feynkit one-loop reduction frontend.

Run in the community host, with its Feynkit and OneLoopMaster bindings enabled:

    SYMBOLICA_HIDE_BANNER=1 pytest python/tests/test_oneloopreduce.py
"""

import math

import pytest
from symbolica import E, Expression, S
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import oneloop


def bubble_family(mass, invariant):
    dimension, loop, external = S("reducer_ffi::D", "reducer_ffi::ell", "reducer_ffi::p")
    kin = hep.Kinematics(dimension, momenta=[loop, external]).with_scalar_product(
        external, external, invariant,
    )
    return hep.IntegralFamily([loop], [external], [
        kin.scalar_product(loop, loop) - mass,
        kin.scalar_product(loop + external, loop + external) - mass,
    ], kinematics=kin)


def test_the_module_has_no_duplicate_input_model():
    assert oneloop.__name__ == "symbolica.community.hepkit.oneloop"
    assert oneloop.Reduction.__module__ == "symbolica.community.hepkit.oneloop"
    assert oneloop.MasterIntegral.__module__ == "symbolica.community.hepkit.oneloop"
    assert not hasattr(oneloop, "Propagator")
    assert not hasattr(oneloop, "IntegralFamily")


def test_massless_bubble_reduces_to_one_master():
    invariant = S("reducer_ffi::s")
    family = bubble_family(E("0"), invariant)
    reduction = oneloop.reduce(family, [1, 1]).simplify()
    ((coefficient, master),) = reduction.terms
    assert (master.kind, master.head) == ("bubble", "B0")
    assert master.arguments == [invariant, E("0"), E("0")]
    assert coefficient == E("1")
    assert reduction.dimension == family.kinematics.dimension


def test_feynkit_numerator_crosses_the_boundary():
    mass, invariant = S("reducer_ffi::m2", "reducer_ffi::s")
    family = bubble_family(mass, invariant)
    numerator = family.kinematics.scalar_product(family.loop_momenta[0], family.external_momenta[0])
    reduction = oneloop.reduce(family, [1, 1], numerator=numerator)
    expected = -invariant / 2 * oneloop.B0(invariant, mass, mass, 1)
    assert isinstance(reduction.to_expression(), Expression)
    assert (reduction.to_expression() - expected).together() == E("0")


def test_reducer_masters_are_primitive_native_hook_symbols():
    mass, invariant, scale = S("reducer_ffi::m2", "reducer_ffi::s", "reducer_ffi::mu2")
    reduction = oneloop.reduce(bubble_family(mass, invariant), [1, 1])
    ((coefficient, master),) = reduction.terms
    canonical = master.to_expression(scale)
    assert canonical == oneloop.B0(invariant, mass, mass, scale)
    assert reduction.to_expression(scale) == coefficient * canonical
    assert oneloop.master_coefficients(canonical) == [
        oneloop.B0(tag, invariant, mass, mass, scale)
        for tag in (0, -1, -2)
    ]


def test_dimension_and_native_hook_finite_part_are_preserved():
    mass, scale, dimension, loop = S(
        "reducer_ffi::m2", "reducer_ffi::mu2", "another_dimension::D", "another_loop::ell",
    )
    kin = hep.Kinematics(dimension, momenta=[loop])
    family = hep.IntegralFamily([loop], [], [kin.scalar_product(loop, loop) - mass], kinematics=kin)
    reduction = oneloop.reduce(family, [2]).simplify()
    assert reduction.dimension == dimension
    expected = (dimension - 2) / (2 * mass) * oneloop.A0(mass, 1)
    assert (reduction.to_expression() - expected).expand() == E("0")
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=scale)
    values = [coefficient.evaluate({mass: 2 + 0j, scale: 3 + 0j}) for coefficient in coefficients]
    assert values == pytest.approx([-math.log(2 / 3), 1, 0], abs=1e-12)


def test_wrong_power_count_raises():
    with pytest.raises(ValueError):
        oneloop.reduce(bubble_family(E("0"), E("-3")), [1])


def test_the_recursion_guard_raises_instead_of_aborting():
    with pytest.raises(ValueError):
        oneloop.reduce(bubble_family(E("0"), E("-3")), [40, 1])
