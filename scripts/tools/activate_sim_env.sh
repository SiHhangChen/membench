#!/usr/bin/env bash
# Source this file to use the repository-local headless simulation environment.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Use: source scripts/tools/activate_sim_env.sh" >&2
    exit 1
fi

_membench_sim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "${_membench_sim_root}/.venv-robocasa/bin/python" ]]; then
    echo "Simulation environment is missing: ${_membench_sim_root}/.venv-robocasa" >&2
    unset _membench_sim_root
    return 1
fi

source "${_membench_sim_root}/.venv-robocasa/bin/activate"
export ROBOCASA_PYTHON="${_membench_sim_root}/.venv-robocasa/bin/python"
export MPLIB_PYTHON="${_membench_sim_root}/.venv-mplib/bin/python"
export MEMBENCH_MPLIB_PYTHON="${MPLIB_PYTHON}"
export MANISKILL_PANDA_DIR="${_membench_sim_root}/outputs/test_env_setup/maniskill/mani_skill/assets/robots/panda"
export MEMBENCH_MANISKILL_PANDA_DIR="${MANISKILL_PANDA_DIR}"
export PYTHONPATH="${_membench_sim_root}:${_membench_sim_root}/robosuite:${_membench_sim_root}/robocasa${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MEMBENCH_RENDER_GPU:-0}"
export MPLCONFIGDIR="${_membench_sim_root}/outputs/test_env_setup/cache/matplotlib"
export NUMBA_CACHE_DIR="${_membench_sim_root}/outputs/test_env_setup/cache/numba"
export XDG_CACHE_HOME="${_membench_sim_root}/outputs/test_env_setup/cache"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

unset _membench_sim_root
