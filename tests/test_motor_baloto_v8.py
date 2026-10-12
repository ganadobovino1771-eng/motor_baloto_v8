import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'motor_baloto_v8', ROOT / 'motor_baloto_v8.py'
)
motor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motor)


def historico_sintetico(cantidad=104):
    filas = []
    for sorteo in range(1, cantidad + 1):
        inicio = (sorteo - 1) % 43 + 1
        numeros = [((inicio + desplazamiento - 1) % 43) + 1 for desplazamiento in range(5)]
        filas.append({
            'Sorteo': sorteo,
            'Fecha': '01 De Enero De 2026',
            'N1': numeros[0], 'N2': numeros[1], 'N3': numeros[2],
            'N4': numeros[3], 'N5': numeros[4],
            'Superbalota': (sorteo - 1) % 16 + 1,
            'Unnamed: extra': '$6.000' if sorteo == cantidad else None,
        })
    return pd.DataFrame(filas)


class MotorBalotoRobustezTests(unittest.TestCase):
    def test_escritura_csv_segura_conserva_backup(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as carpeta:
            archivo = Path(carpeta) / 'datos.csv'
            archivo.write_text('anterior\n', encoding='utf-8')
            motor.escribir_csv_seguro(pd.DataFrame({'valor': [1]}), str(archivo))

            self.assertEqual(Path(str(archivo) + '.bak').read_text(encoding='utf-8'), 'anterior\n')
            self.assertEqual(pd.read_csv(archivo, sep=';').iloc[0]['valor'], 1)
            self.assertFalse(list(Path(carpeta).glob('*.tmp')))

    def test_historico_invalido_aborta_sin_escribir_archivos(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as carpeta:
            raiz = Path(carpeta)
            invalido = historico_sintetico(3)
            invalido.loc[2, 'N5'] = invalido.loc[2, 'N4']
            invalido.to_csv(raiz / 'Baloto_Historico_Perfecto.csv', sep=';', index=False)
            registro = raiz / 'Registro_Inversiones.csv'
            auditoria = raiz / 'Auditoria_Filtros.csv'
            registro.write_text('registro intacto\n', encoding='utf-8')
            auditoria.write_text('auditoria intacta\n', encoding='utf-8')

            cwd_original = os.getcwd()
            try:
                os.chdir(raiz)
                motor.acum_baloto = 30000
                motor.acum_revancha = 30000
                salida = io.StringIO()
                with contextlib.redirect_stdout(salida):
                    motor.ejecutar_sistema_profesional()
            finally:
                os.chdir(cwd_original)

            self.assertIn('Historial inválido', salida.getvalue())
            self.assertIn('cinco números distintos', salida.getvalue())
            self.assertEqual(registro.read_text(encoding='utf-8'), 'registro intacto\n')
            self.assertEqual(auditoria.read_text(encoding='utf-8'), 'auditoria intacta\n')
            self.assertFalse((raiz / 'Reportes').exists())

            (raiz / 'Baloto_Historico_Perfecto.csv').write_text('', encoding='utf-8')
            salida_vacia = io.StringIO()
            try:
                os.chdir(raiz)
                with contextlib.redirect_stdout(salida_vacia):
                    motor.ejecutar_sistema_profesional()
            finally:
                os.chdir(cwd_original)
            self.assertIn('el archivo está vacío', salida_vacia.getvalue())
            self.assertEqual(registro.read_text(encoding='utf-8'), 'registro intacto\n')
            self.assertEqual(auditoria.read_text(encoding='utf-8'), 'auditoria intacta\n')

    def test_sorteo_duplicado_no_agrega_filas_y_avisa(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as carpeta:
            raiz = Path(carpeta)
            historico_sintetico().to_csv(
                raiz / 'Baloto_Historico_Perfecto.csv', sep=';', index=False
            )
            columnas_registro = [
                'Fecha_Ejecucion', 'Sorteo_Objetivo', 'Jugada', 'SB', 'Inversion',
                'Aciertos_Post', 'Premio', 'Alerta_SET', 'Correlacion_SET',
                'Cambios_Top5', 'Estado_SET', 'Version_Motor',
            ]
            registro = pd.DataFrame([{
                'Fecha_Ejecucion': '01/01/2026', 'Sorteo_Objetivo': 105,
                'Jugada': '[4, 7, 10, 27, 41]', 'SB': 16, 'Inversion': 9000,
                'Aciertos_Post': 'Pendiente', 'Premio': 0,
                'Alerta_SET': 'NO', 'Correlacion_SET': 0.1,
                'Cambios_Top5': 0, 'Estado_SET': 'ESTABLE', 'Version_Motor': 'v8.8C',
            }], columns=columnas_registro)
            registro.to_csv(raiz / 'Registro_Inversiones.csv', sep=';', index=False)
            pd.DataFrame([{'Sorteo_Objetivo': 105, 'Pool_Inicial': 1000000}]).to_csv(
                raiz / 'Auditoria_Filtros.csv', sep=';', index=False
            )

            eleccion_original = motor.np.random.choice

            def eleccion_prueba(valores, size=None, replace=True):
                if len(valores) == 43:
                    return np.array([4, 7, 10, 27, 41])
                if len(valores) == 16:
                    return 16
                return eleccion_original(valores, size=size, replace=replace)

            cwd_original = os.getcwd()
            try:
                os.chdir(raiz)
                motor.acum_baloto = 30000
                motor.acum_revancha = 30000
                motor.np.random.choice = eleccion_prueba
                salida = io.StringIO()
                with patch('builtins.input', return_value='s'), contextlib.redirect_stdout(salida):
                    motor.ejecutar_sistema_profesional()
            finally:
                motor.np.random.choice = eleccion_original
                os.chdir(cwd_original)

            texto = salida.getvalue()
            self.assertIn('⚠ El sorteo 105 ya estaba registrado; no se agregó fila', texto)
            self.assertEqual(texto.count('⚠ El sorteo 105 ya estaba registrado; no se agregó fila'), 2)
            self.assertNotIn('✔ Alerta SET persistida', texto)
            self.assertEqual(len(pd.read_csv(raiz / 'Registro_Inversiones.csv', sep=';')), 1)
            self.assertEqual(len(pd.read_csv(raiz / 'Auditoria_Filtros.csv', sep=';')), 1)

    def test_aciertos_vacio_se_descarta_del_bloque(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as carpeta:
            archivo = Path(carpeta) / 'Registro_Inversiones.csv'
            pd.DataFrame({
                'Sorteo_Objetivo': [1, 2, 3, 4, 5],
                'Aciertos_Post': ['1N + 0SB', '2N + 0SB', '', '0N + 1SB', '3N + 0SB'],
                'Estado_SET': ['ESTABLE'] * 5,
            }).to_csv(archivo, sep=';', index=False)

            bloque = motor.analisis_trazabilidad_bloque(str(archivo))

            self.assertEqual(bloque['sorteos_bloque'], [1, 2, 4, 5])
            self.assertEqual(bloque['detalle_aciertos'], [1, 2, 0, 3])
            self.assertEqual(bloque['total_aciertos'], 6)

            pd.DataFrame({
                'Sorteo_Objetivo': [1, 2, 3, 4],
                'Aciertos_Post': ['1N + 0SB', '', 'sin dato', '2N + 0SB'],
            }).to_csv(archivo, sep=';', index=False)
            self.assertIsNone(motor.analisis_trazabilidad_bloque(str(archivo)))

    def test_importar_modulo_no_ejecuta_el_motor(self):
        spec = importlib.util.spec_from_file_location(
            'motor_baloto_importado', ROOT / 'motor_baloto_v8.py'
        )
        modulo = importlib.util.module_from_spec(spec)
        salida = io.StringIO()
        with patch('builtins.input', side_effect=AssertionError('no debe preguntar')):
            with contextlib.redirect_stdout(salida):
                spec.loader.exec_module(modulo)
        self.assertEqual(salida.getvalue(), '')
        self.assertTrue(callable(modulo.ejecutar_sistema_profesional))


if __name__ == '__main__':
    unittest.main()

