# 0. limpiar
Remove-Item salida\* -Exclude .gitkeep

# 1. marcar el arco (cuadro ~180, click en las 4 esquinas interiores)
python src\marcar_esquinas.py videos\penal1.mp4

# 2. correr el pipeline (usa --hasta 330, no el valor que sugiere el script)
python src\pipeline_balon.py videos\penal1.mp4 --ref 180 --esquinas 37,399 128,399 128,487 37,490 --hasta 330 --salida salida

# 3. mostrar resultados
start salida\demo_zoom_lento.mp4
start salida\trayectoria_balon.png