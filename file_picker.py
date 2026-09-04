"""
Módulo para seleccionar archivos de video mediante el explorador de archivos nativo de Windows (Microsoft File Picker).
"""
import sys
import tkinter as tk
from tkinter import filedialog
from typing import Optional

def select_video_file() -> Optional[str]:
    """
    Abre la ventana nativa de selección de archivos de Windows y devuelve la ruta seleccionada.
    """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    root.attributes('-topmost', True)  # Asegurar que aparezca al frente

    # Filtros de archivos de video comunes
    filetypes = [
        ("Archivos de Video", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv;*.webm;*.m4v"),
        ("MP4 Video (*.mp4)", "*.mp4"),
        ("AVI Video (*.avi)", "*.avi"),
        ("QuickTime (*.mov)", "*.mov"),
        ("Matroska (*.mkv)", "*.mkv"),
        ("Todos los archivos", "*.*")
    ]

    selected_file = filedialog.askopenfilename(
        title="Selecciona un archivo de video para analizar",
        filetypes=filetypes
    )

    root.destroy()

    if selected_file:
        return selected_file
    return None

if __name__ == "__main__":
    video = select_video_file()
    print("Archivo seleccionado:", video)
