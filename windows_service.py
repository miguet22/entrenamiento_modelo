"""Instala y ejecuta el servidor como servicio nativo de Windows."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import ctypes
import multiprocessing
from pathlib import Path
import os
import sys
from threading import Event
import time

import servicemanager
import win32service
import win32serviceutil

ROOT = Path(__file__).resolve().parent
SERVICE_NAME = 'DeteccionVideoIA'


class VideoDetectionService(win32serviceutil.ServiceFramework):
    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = 'Deteccion de video con IA'
    _svc_description_ = 'API local y WebSocket de deteccion de vehiculos, cascos y choques.'

    def __init__(self, args):
        super().__init__(args)
        self.stopping = Event()
        self.server = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING, waitHint=30000)
        self.stopping.set()
        if self.server is not None:
            self.server.should_exit = True

    def SvcShutdown(self):
        self.SvcStop()

    def SvcDoRun(self):
        os.chdir(ROOT)
        logs = ROOT / 'service_data'
        logs.mkdir(exist_ok=True)
        with (logs / 'service.log').open('a', encoding='utf-8', buffering=1) as output:
            with redirect_stdout(output), redirect_stderr(output):
                from run_service import build_server
                self.server = build_server()
                if self.stopping.is_set():
                    return
                self.server.run()
                if not self.stopping.is_set():
                    raise RuntimeError('El servidor se detuvo inesperadamente; revisar service_data/service.log')


def wait_stopped():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if win32serviceutil.QueryServiceStatus(SERVICE_NAME)[1] == win32service.SERVICE_STOPPED:
            return
        time.sleep(0.5)
    raise RuntimeError('El servicio no se detuvo en 30 segundos')


def manage(command):
    if command != 'status' and not ctypes.windll.shell32.IsUserAnAdmin():
        raise PermissionError('Este comando requiere una terminal ejecutada como administrador')
    if command == 'install':
        executable_args = f'"{Path(__file__).resolve()}" run'
        win32serviceutil.InstallService(
            'windows_service.VideoDetectionService', SERVICE_NAME,
            VideoDetectionService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START,
            exeName=sys.executable, exeArgs=executable_args,
            description=VideoDetectionService._svc_description_, delayedstart=True)
        manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
        try:
            service = win32service.OpenService(manager, SERVICE_NAME, win32service.SERVICE_CHANGE_CONFIG)
            try:
                win32service.ChangeServiceConfig2(service, win32service.SERVICE_CONFIG_FAILURE_ACTIONS,
                    dict(ResetPeriod=86400, RebootMsg='', Command='',
                         Actions=[(win32service.SC_ACTION_RESTART, 5000)] * 3))
                win32service.ChangeServiceConfig2(service,
                    win32service.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, True)
            finally:
                win32service.CloseServiceHandle(service)
        finally:
            win32service.CloseServiceHandle(manager)
        win32serviceutil.StartService(SERVICE_NAME)
        print('Servicio instalado con inicio automatico. Iniciando API...')
    elif command == 'start':
        win32serviceutil.StartService(SERVICE_NAME)
        print('Inicio solicitado')
    elif command in ('stop', 'remove'):
        if win32serviceutil.QueryServiceStatus(SERVICE_NAME)[1] != win32service.SERVICE_STOPPED:
            win32serviceutil.StopService(SERVICE_NAME)
            wait_stopped()
        if command == 'remove':
            win32serviceutil.RemoveService(SERVICE_NAME)
            print('Servicio desinstalado')
        else:
            print('Servicio detenido')
    else:
        state = win32serviceutil.QueryServiceStatus(SERVICE_NAME)[1]
        print(SERVICE_NAME, {1: 'detenido', 2: 'iniciando', 3: 'deteniendo',
                              4: 'en ejecucion'}.get(state, str(state)))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['install', 'start', 'stop', 'remove', 'status', 'run'])
    command = parser.parse_args().command
    if command == 'run':
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(VideoDetectionService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        try:
            manage(command)
        except Exception as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            sys.exit(1)
