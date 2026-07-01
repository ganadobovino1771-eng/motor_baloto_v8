import pandas as pd
import numpy as np
from datetime import datetime
import os
import ast
import re

# ==========================================
# CONFIGURACIÓN SEMANAL (Actualizar el Jueves)
# ==========================================
acum_baloto   = 40000   # Millones
acum_revancha = 4000   # Millones
costo_ticket  = 9000    # Baloto + Revancha

# ==========================================
# VENTANA DE ANÁLISIS
# v8.0: Se usan los últimos 52 sorteos (~1 año)
# para calcular números calientes y Superbalota
# ==========================================
VENTANA_CALIENTES = 52

# ==========================================
# MÓDULOS DE AUDITORÍA
# Informativos — NO modifican la jugada.
# Principio: el motor no cambia,
# la inteligencia que lo rodea crece.
# ==========================================

def auditar_estabilidad_balotas(df_hist, ventana=52):
    """
    MÓDULO 1 — ALERTA DE SET (v8.3)
    Detecta posibles cambios en las balotas del bombo
    comparando dos ventanas consecutivas de 52 sorteos.
    Usa Top5 + Correlación de frecuencias completa.
    """
    if len(df_hist) < ventana * 2:
        return None

    ventana_reciente = df_hist.tail(ventana)
    ventana_anterior = df_hist.iloc[-(ventana * 2):-ventana]

    def frecuencias(df_v):
        nums = pd.concat([df_v['N1'], df_v['N2'], df_v['N3'],
                          df_v['N4'], df_v['N5']])
        freq = nums.value_counts()
        full = pd.Series(0, index=range(1, 44))
        full.update(freq)
        return full

    freq_rec     = frecuencias(ventana_reciente)
    freq_ant     = frecuencias(ventana_anterior)
    top5_rec     = set(freq_rec.nlargest(5).index.tolist())
    top5_ant     = set(freq_ant.nlargest(5).index.tolist())
    interseccion = len(top5_rec & top5_ant)   # cuántos se mantuvieron
    cambios_top5 = 5 - interseccion           # cuántos cambiaron
    correlacion  = freq_rec.corr(freq_ant)

    estado_top5 = "ESTABLE"   if cambios_top5 <= 2 else \
                  "MONITOREAR" if cambios_top5 == 3  else "ALERTA"

    estado_corr = "ESTABLE"   if correlacion > 0.80 else \
                  "MONITOREAR" if correlacion >= 0.60 else "RUPTURA"

    if "ALERTA" in estado_top5 or "RUPTURA" in estado_corr:
        veredicto         = "ALERTA"
        veredicto_display = "🚨 POSIBLE CAMBIO DE BALOTAS — Revisar con Baloto"
    elif "MONITOREAR" in estado_top5 or "MONITOREAR" in estado_corr:
        veredicto         = "MONITOREAR"
        veredicto_display = "⚠️  DRIFT DETECTADO — Mantener vigilancia"
    else:
        veredicto         = "ESTABLE"
        veredicto_display = "✅ SISTEMA ESTABLE — Sin anomalías"

    return {
        'cambios_top5'     : cambios_top5,
        'estado_top5'      : estado_top5,
        'correlacion'      : round(correlacion, 4),
        'estado_corr'      : estado_corr,
        'veredicto'        : veredicto,
        'veredicto_display': veredicto_display,
        'top5_rec'         : sorted(top5_rec),
        'top5_ant'         : sorted(top5_ant)
    }


def auditar_zona_suma(df_hist, ventana=52):
    """
    MÓDULO 2 — ZONA CALIENTE DE SUMA (v8.3)
    Analiza distribución de sumas en 3 zonas
    dentro del rango 95-125 ya validado.
    Solo informativo — NO cambia el filtro.
    """
    df_v  = df_hist.tail(ventana)
    sumas = (df_v['N1'] + df_v['N2'] + df_v['N3'] +
             df_v['N4'] + df_v['N5'])
    sumas_validas = sumas[(sumas >= 95) & (sumas <= 125)]

    if len(sumas_validas) == 0:
        return None

    bajo  = ((sumas_validas >= 95)  & (sumas_validas <= 105)).sum()
    medio = ((sumas_validas >= 106) & (sumas_validas <= 115)).sum()
    alto  = ((sumas_validas >= 116) & (sumas_validas <= 125)).sum()
    total = len(sumas_validas)

    pct_bajo  = round(bajo  / total * 100, 1)
    pct_medio = round(medio / total * 100, 1)
    pct_alto  = round(alto  / total * 100, 1)
    mediana   = round(float(sumas_validas.median()), 1)
    promedio  = round(float(sumas_validas.mean()), 1)

    if medio >= bajo and medio >= alto:
        zona_dominante = f"MEDIA (106-115) — {pct_medio}% de sorteos"
    elif bajo >= medio and bajo >= alto:
        zona_dominante = f"BAJA (95-105) — {pct_bajo}% de sorteos"
    else:
        zona_dominante = f"ALTA (116-125) — {pct_alto}% de sorteos"

    return {
        'pct_bajo'      : pct_bajo,
        'pct_medio'     : pct_medio,
        'pct_alto'      : pct_alto,
        'mediana'       : mediana,
        'promedio'      : promedio,
        'zona_dominante': zona_dominante
    }


def analisis_trazabilidad_bloque(archivo_log):
    """
    MÓDULO 3 — TRAZABILIDAD ACUMULADA (v8.5)
    Parte 3 del Protocolo — se activa automáticamente
    cuando hay 4 o más registros en Registro_Inversiones.csv.

    Evalúa los últimos 4 sorteos jugados:
    - Aciertos acumulados y promedio
    - Comportamiento de Alertas SET
    - Correlación promedio SET
    - Veredicto del bloque

    AJUSTE v8.5: Umbral ALTO calibrado a >=1.5 aciertos
    promedio (no 2.0) — más realista para Baloto Colombia
    donde 5/43 implica ~1 acierto esperado por sorteo.

    Solo informativo — NO modifica la jugada.
    """
    if not os.path.exists(archivo_log):
        return None

    df = pd.read_csv(archivo_log, sep=';')

    # Solo activar si hay al menos 4 registros completos
    # (Aciertos_Post ya calculados, no Pendiente)
    df_completados = df[df['Aciertos_Post'] != 'Pendiente']
    if len(df_completados) < 4:
        return None

    df_4 = df_completados.tail(4).copy()

    # 1. ACIERTOS ACUMULADOS
    # v8.6 — Parsing blindado con regex
    # Tolerante a variaciones de formato:
    # "2N + 1SB", "2N+1SB", "2 N + 1 SB"
    total_aciertos = 0
    detalles       = []
    for val in df_4['Aciertos_Post']:
        if isinstance(val, str):
            match = re.search(r'(\d+)N', val)
            n = int(match.group(1)) if match else 0
        else:
            n = 0
        total_aciertos += n
        detalles.append(n)

    promedio_aciertos = round(total_aciertos / 4, 2)

    # Umbral calibrado para Baloto Colombia
    # E[aciertos] = 5/43 * 5 ≈ 0.58 por sorteo
    # ≥1.5 promedio = rendimiento alto real
    if promedio_aciertos >= 1.5:
        nivel_efectividad = "ALTO"
    elif promedio_aciertos >= 0.75:
        nivel_efectividad = "MEDIO"
    else:
        nivel_efectividad = "BAJO"

    # 2. ALERTA SET — FRECUENCIA EN EL BLOQUE
    if 'Estado_SET' in df_4.columns:
        estados = df_4['Estado_SET'].tolist()
        alertas   = sum(1 for e in estados if str(e) == 'ALERTA')
        monitoreo = sum(1 for e in estados if str(e) == 'MONITOREAR')
        estable   = sum(1 for e in estados if str(e) == 'ESTABLE')
    else:
        alertas = monitoreo = estable = 0

    # 3. CORRELACIÓN PROMEDIO SET
    if 'Correlacion_SET' in df_4.columns:
        corr_vals = pd.to_numeric(df_4['Correlacion_SET'], errors='coerce')
        corr_vals = corr_vals.dropna()
        corr_prom = round(float(corr_vals.mean()), 4) if len(corr_vals) > 0 else None
    else:
        corr_prom = None

    # 4. VEREDICTO DEL BLOQUE
    # Criterio: calidad de señal, no solo cantidad
    if alertas >= 2:
        veredicto = "🚨 INESTABILIDAD DETECTADA — Revisar comportamiento del bombo"
    elif alertas == 1 and monitoreo >= 1:
        veredicto = "⚠️  SEÑAL MIXTA — Mantener vigilancia activa"
    elif monitoreo >= 2:
        veredicto = "⚠️  VARIACIÓN PRESENTE — Continuar monitoreo"
    else:
        veredicto = "✅ SISTEMA ESTABLE — Comportamiento normal"

    # 5. ALERTA DE AJUSTE (Protocolo v1.2)
    alerta_ajuste = None
    if alertas >= 3:
        alerta_ajuste = "🚨 ALERTA DE AJUSTE — 3+ semanas con señal SET. Revisar en próxima revisión trimestral."

    return {
        'sorteos_bloque'    : df_4['Sorteo_Objetivo'].tolist(),
        'total_aciertos'    : total_aciertos,
        'detalle_aciertos'  : detalles,
        'promedio_aciertos' : promedio_aciertos,
        'nivel_efectividad' : nivel_efectividad,
        'alertas'           : alertas,
        'monitoreo'         : monitoreo,
        'estable'           : estable,
        'corr_promedio'     : corr_prom,
        'veredicto'         : veredicto,
        'alerta_ajuste'     : alerta_ajuste
    }


def ejecutar_sistema_profesional():

    # 0. CONTROL DE UMBRAL (Protección de Capital)
    if acum_baloto < 15000 and acum_revancha < 15000:
        print("\n" + "!"*45)
        print("  ALERTA: PROTOCOLO DE AHORRO ACTIVADO")
        print(f"  Acumulados: B:{acum_baloto}M / R:{acum_revancha}M")
        print("  Ambos < 15.000M — NO SE GENERA JUGADA.")
        print("!"*45)
        return

    archivo_hist = 'Baloto_Historico_Perfecto.csv'
    archivo_log  = 'Registro_Inversiones.csv'

    if not os.path.exists(archivo_hist):
        print(f"ERROR: No se encontró {archivo_hist}")
        return

    # 1. CARGAR HISTORIAL Y SEMILLA DETERMINISTA
    df_hist = pd.read_csv(archivo_hist, sep=';')
    ultimo  = df_hist.iloc[-1]

    print("\n" + "─"*45)
    print("  VERIFICACIÓN DE HISTORIAL")
    print("─"*45)
    print(f"  Último sorteo en CSV : {int(ultimo['Sorteo'])}")
    print(f"  Fecha                : {ultimo['Fecha']}")
    print(f"  Números              : {[int(ultimo[f'N{i}']) for i in range(1,6)]}  SB: {int(ultimo['Superbalota'])}")
    print("─"*45)
    
    # ── BUG 2 CORREGIDO: Detectar si sorteo objetivo es sábado ──
    proximo_sorteo_num = int(ultimo['Sorteo']) + 1
    
    while True:
        print(f"  Sorteo objetivo      : {proximo_sorteo_num}")
        confirmacion = input("  ¿Este sorteo corresponde al SÁBADO? (s/n): ").strip().lower()
        
        if confirmacion == 's':
            print("─"*45)
            break
        elif confirmacion == 'n':
            proximo_sorteo_num += 1
            print("  Avanzando al siguiente sorteo...")
        else:
            print("  ⚠ Responde 's' o 'n'.")

    np.random.seed(proximo_sorteo_num)

    num_espejo = [int(ultimo[f'N{i}']) for i in range(1, 6)]
    sb_espejo  = int(ultimo['Superbalota'])

    # 2. AUDITORÍA DE ROI — BUG 1 CORREGIDO
    if os.path.exists(archivo_log):
        df_log  = pd.read_csv(archivo_log, sep=';')
        cambios = False
        for i, row in df_log.iterrows():
            if row['Aciertos_Post'] == 'Pendiente':
                sorteo_objetivo = int(row['Sorteo_Objetivo'])
                jugada_ant = ast.literal_eval(row['Jugada'])
                sb_ant     = int(row['SB'])
                
                # Buscar el resultado oficial en el historial
                resultado = df_hist[df_hist['Sorteo'] == sorteo_objetivo]
                
                if len(resultado) > 0:
                    # Comparar contra el resultado oficial del sorteo objetivo
                    num_oficial = [int(resultado.iloc[0][f'N{i}']) for i in range(1, 6)]
                    sb_oficial  = int(resultado.iloc[0]['Superbalota'])
                    
                    aciertos   = len(np.intersect1d(jugada_ant, num_oficial))
                    sb_acierto = 1 if sb_ant == sb_oficial else 0
                    df_log.at[i, 'Aciertos_Post'] = f"{aciertos}N + {sb_acierto}SB"
                    cambios = True
                else:
                    # Si no está en el historial, dejar como Pendiente
                    pass
        
        if cambios:
            df_log.to_csv(archivo_log, index=False, sep=';')
            print("\n✔ Auditoría actualizada — recuerda registrar el Premio manualmente.")

    # 3. ANÁLISIS DE TENDENCIAS — ÚLTIMO AÑO (v8.0)
    df_reciente = df_hist.tail(VENTANA_CALIENTES)
    todos_rec   = pd.concat([
        df_reciente['N1'], df_reciente['N2'], df_reciente['N3'],
        df_reciente['N4'], df_reciente['N5']
    ])
    calientes = todos_rec.value_counts().head(5).index.tolist()
    sb_oro    = df_reciente['Superbalota'].value_counts().head(2).index.tolist()

    # 3B. MÓDULOS DE AUDITORÍA — Solo informativos
    alerta_set   = auditar_estabilidad_balotas(df_hist, VENTANA_CALIENTES)
    zona_suma    = auditar_zona_suma(df_hist, VENTANA_CALIENTES)
    trazabilidad = analisis_trazabilidad_bloque(archivo_log)

    # 4. SIMULACIÓN MONTE CARLO — 1.000.000 jugadas
    print(f"\n--- PROCESANDO SORTEO {proximo_sorteo_num} (SÁBADO) ---")
    n    = 1_000_000
    pool = np.array([
        np.random.choice(np.arange(1, 44), 5, replace=False)
        for _ in range(n)
    ])

    # 5. FILTROS — PROTOCOLO v2.3 — MOTOR INTACTO DESDE v8.2
    def consecutivos_ok(c):
        return (np.diff(np.sort(c)) == 1).sum() <= 1

    mask_espejo  = np.array([len(np.intersect1d(c, num_espejo)) == 1 for c in pool])
    mask_suma    = (pool.sum(axis=1) >= 95) & (pool.sum(axis=1) <= 125)
    mask_paridad = ((pool % 2 == 0).sum(axis=1) >= 2) & ((pool % 2 == 0).sum(axis=1) <= 3)
    mask_ev      = (pool > 31).any(axis=1)
    mask_consec  = np.array([consecutivos_ok(c) for c in pool])

    sobrevivientes = pool[mask_espejo & mask_suma & mask_paridad & mask_ev & mask_consec]

    if len(sobrevivientes) == 0:
        print("⚠ Sin sobrevivientes tras los filtros. Revisa el historial.")
        return

    # 6. SELECCIÓN POR PUNTAJE DE CALIENTES — TOP 10% (v8.1)
    scores   = np.array([len(np.intersect1d(c, calientes)) for c in sobrevivientes])
    umbral   = np.percentile(scores, 90)
    top      = sobrevivientes[scores >= umbral]
    ganadora = sorted([int(x) for x in top[np.random.randint(len(top))]])
    sb_final = int(np.random.choice(sb_oro))

    # 7. REGISTRO DE INVERSIÓN — v8.4+
    if alerta_set:
        reg_alerta_set   = 'SI'         if alerta_set['veredicto'] == 'ALERTA'     else \
                           'MONITOREAR'  if alerta_set['veredicto'] == 'MONITOREAR' else 'NO'
        reg_correlacion  = alerta_set['correlacion']
        reg_cambios_top5 = alerta_set['cambios_top5']
        reg_estado_set   = alerta_set['veredicto']
    else:
        reg_alerta_set   = 'SIN_DATOS'
        reg_correlacion  = None
        reg_cambios_top5 = None
        reg_estado_set   = 'SIN_DATOS'

    nueva_fila = pd.DataFrame([{
        'Fecha_Ejecucion' : datetime.now().strftime('%d/%m/%Y'),
        'Sorteo_Objetivo' : proximo_sorteo_num,
        'Jugada'          : str(ganadora),
        'SB'              : sb_final,
        'Inversion'       : costo_ticket,
        'Aciertos_Post'   : 'Pendiente',
        'Premio'          : 0,
        'Alerta_SET'      : reg_alerta_set,
        'Correlacion_SET' : reg_correlacion,
        'Cambios_Top5'    : reg_cambios_top5,
        'Estado_SET'      : reg_estado_set
    }])

    if not os.path.exists(archivo_log):
        nueva_fila.to_csv(archivo_log, index=False, sep=';')
    else:
        df_existente = pd.read_csv(archivo_log, sep=';')
        for col in ['Alerta_SET', 'Correlacion_SET', 'Cambios_Top5', 'Estado_SET']:
            if col not in df_existente.columns:
                df_existente[col] = 'PREVIO_v8.4'
        if proximo_sorteo_num not in df_existente['Sorteo_Objetivo'].values:
            df_existente = pd.concat([df_existente, nueva_fila], ignore_index=True)
        df_existente.to_csv(archivo_log, index=False, sep=';')

    # 8. BOLETÍN FINAL
    pares       = sum(1 for x in ganadora if x % 2 == 0)
    impares     = 5 - pares
    suma_jugada = sum(ganadora)

    if zona_suma:
        if 95 <= suma_jugada <= 105:
            zona_jugada = f"BAJA (95-105) — zona {zona_suma['pct_bajo']}% histórico"
        elif 106 <= suma_jugada <= 115:
            zona_jugada = f"MEDIA (106-115) — zona {zona_suma['pct_medio']}% histórico"
        else:
            zona_jugada = f"ALTA (116-125) — zona {zona_suma['pct_alto']}% histórico"
    else:
        zona_jugada = "—"

    print("\n" + "█"*45)
    print(f"        JUGADA MAESTRA — SORTEO {proximo_sorteo_num}")
    print(f"        (Versión 8.7 — Corrección de Bugs)")
    print("█"*45)
    print(f"  JUGADA  : {ganadora}")
    print(f"  SB      : {sb_final}")
    print(f"  SUMA    : {suma_jugada}  [{zona_jugada}]")
    print(f"  PARIDAD : {pares} Pares / {impares} Impares")
    print(f"  ESPEJO  : {num_espejo} → repite 1 número ✔")
    print("─"*45)
    print(f"  Calientes usados (último año): {calientes}")
    print(f"  SB candidatos (último año)   : {sb_oro}")
    print("─"*45)
    print(f"  Acumulado Baloto  : ${acum_baloto:,}M")
    print(f"  Acumulado Revancha: ${acum_revancha:,}M")
    print(f"  Inversión         : ${costo_ticket:,}")
    print("─"*45)
    print("  AUDITORÍA INTELIGENTE v8.7")
    print("─"*45)

    if alerta_set:
        icon_top5 = "✅" if alerta_set['estado_top5'] == "ESTABLE"    else \
                    "⚠️ " if alerta_set['estado_top5'] == "MONITOREAR" else "🚨"
        icon_corr = "✅" if alerta_set['estado_corr'] == "ESTABLE"    else \
                    "⚠️ " if alerta_set['estado_corr'] == "MONITOREAR" else "🚨"
        print(f"  ESTABILIDAD DE BALOTAS")
        print(f"  Top5 actual     : {alerta_set['top5_rec']}")
        print(f"  Top5 anterior   : {alerta_set['top5_ant']}")
        print(f"  Cambios Top5    : {alerta_set['cambios_top5']}  {icon_top5} {alerta_set['estado_top5']}")
        print(f"  Correlación     : {alerta_set['correlacion']}  {icon_corr} {alerta_set['estado_corr']}")
        print(f"  Veredicto SET   : {alerta_set['veredicto_display']}")
        print(f"  ✔ Alerta persistida en Registro_Inversiones.csv")
    else:
        print("  ESTABILIDAD: Historial insuficiente (requiere 104+ sorteos)")

    print("─"*45)
    if zona_suma:
        print(f"  DISTRIBUCIÓN DE SUMAS (últimos {VENTANA_CALIENTES} sorteos)")
        print(f"  Zona Baja  (95-105) : {zona_suma['pct_bajo']}%")
        print(f"  Zona Media (106-115): {zona_suma['pct_medio']}%")
        print(f"  Zona Alta  (116-125): {zona_suma['pct_alto']}%")
        print(f"  Mediana histórica   : {zona_suma['mediana']}  |  Promedio: {zona_suma['promedio']}")
        print(f"  Zona dominante      : {zona_suma['zona_dominante']}")
        print(f"  Nuestra jugada cae en: {zona_jugada}")

    # ── PARTE 3: TRAZABILIDAD ACUMULADA (automática cada 4 sorteos) ──
    if trazabilidad:
        print("─"*45)
        print("  ★ PARTE 3 — TRAZABILIDAD ACUMULADA")
        print(f"  Sorteos del bloque  : {trazabilidad['sorteos_bloque']}")
        print("─"*45)
        print(f"  Aciertos por sorteo : {trazabilidad['detalle_aciertos']}")
        print(f"  Total aciertos      : {trazabilidad['total_aciertos']}")
        print(f"  Promedio            : {trazabilidad['promedio_aciertos']}  → {trazabilidad['nivel_efectividad']}")
        print("─"*45)
        print(f"  SET ALERTA          : {trazabilidad['alertas']} de 4 sorteos")
        print(f"  SET MONITOREAR      : {trazabilidad['monitoreo']} de 4 sorteos")
        print(f"  SET ESTABLE         : {trazabilidad['estable']} de 4 sorteos")
        if trazabilidad['corr_promedio'] is not None:
            print(f"  Correlación prom SET: {trazabilidad['corr_promedio']}")
        print("─"*45)
        print(f"  Veredicto del bloque: {trazabilidad['veredicto']}")
        if trazabilidad['alerta_ajuste']:
            print(f"  {trazabilidad['alerta_ajuste']}")

    print("█"*45)
    print("  ✔ Registro actualizado en Registro_Inversiones.csv")
    print("  ✔ Alerta SET persistida para análisis histórico.")
    print("  ⚠ Recuerda ingresar el Premio manualmente tras el sorteo.")
    print("  ℹ Módulos de auditoría son informativos — no afectan la jugada.")

ejecutar_sistema_profesional()
