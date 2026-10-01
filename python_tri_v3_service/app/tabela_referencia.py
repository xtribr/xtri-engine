"""
Gerenciador da tabela de referência TRI oficial do ENEM.

Carrega os dados de referência (tri_min, tri_med, tri_max) por área e número
de acertos, usados para ancoragem das notas em ambos os motores.
"""

import json
import os
from typing import Dict, Optional

# Limites máximos EMPIRICOS observados nos microdados ENEM 2024
# (fonte: ENEM 2024 - dificuldades.pdf, tabela oficial INEP por acertos)
#
# Valores reais máximos observados em 2024 (no acerto máximo de cada área):
#   LC INGLÊS: 45 acertos → 795.8 (18 alunos no topo)
#   LC ESPANHOL: 45 acertos → 795.8 (2 alunos no topo)
#   CH:        45 acertos → 819.7 (111 alunos no topo)
#   CN:        44 acertos → 867.2 (17 alunos no topo, sem dado para 45)
#   MT:        45 acertos → 961.9 (506 alunos no topo)
#
# Tetos abaixo são round-down conservadores em relação ao máximo empírico,
# para evitar inflação a partir de calibrações locais. Valores anteriores
# (LC=800, CH=835, CN=875, MT=990) tinham buffer excessivo acima do real.
# Atualizado em 2026-04-07 contra o PDF oficial 2024.
TRI_MAXIMA_OFICIAL = {
    'LC': 790.0,   # 2024 real: 795.8 — round-down para 790
    'CH': 815.0,   # 2024 real: 819.7 — round-down para 815
    'CN': 865.0,   # 2024 real: 867.2 — round-down para 865
    'MT': 960.0,   # 2024 real: 961.9 — round-down para 960
}

# Percentis 99.9 reais (top 0.1% dos alunos) — útil para sanity check
# de calibrações locais agressivas
TRI_P999_REAL = {
    'LC': 720.0,
    'CH': 770.0,
    'CN': 780.0,
    'MT': 920.0,
}

# Limites mínimos oficiais do ENEM
TRI_MINIMA_OFICIAL = {
    'LC': 294.0,
    'CH': 284.0,
    'CN': 308.0,
    'MT': 334.0,
}


class TabelaReferenciaTRI:
    """
    Gerenciador de tabela de referência TRI oficial.

    Estrutura interna (lookup):
    {
        'CH': {
            0: {'tri_min': 329.8, 'tri_med': 329.8, 'tri_max': 329.8},
            1: {'tri_min': 326.2, 'tri_med': 338.3, 'tri_max': 363.3},
            ...
            45: {'tri_min': ..., 'tri_med': ..., 'tri_max': ...}
        },
        'CN': {...},
        'LC': {...},
        'MT': {...}
    }
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Carrega tabela de referência a partir do JSON oficial.

        Args:
            json_path: Caminho para 'tri_tabela_referencia_oficial.json'.
                       Se None, usa o caminho padrão dentro de app/referencias/.
        """
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'referencias',
                'tri_tabela_referencia_oficial.json'
            )

        with open(json_path, 'r') as f:
            raw = json.load(f)

        # Construir lookup com chaves inteiras
        self.lookup: Dict[str, Dict[int, Dict[str, float]]] = {}
        for area, acertos_dict in raw.items():
            self.lookup[area] = {}
            for acertos_str, valores in acertos_dict.items():
                acertos = int(acertos_str)
                self.lookup[area][acertos] = {
                    'tri_min': round(valores['tri_min'], 1),
                    'tri_med': round(valores['tri_med'], 1),
                    'tri_max': round(valores['tri_max'], 1),
                }

        # Áreas disponíveis
        self.areas = list(self.lookup.keys())
        self.ano: Optional[int] = None
        self.quantis: Dict[str, Dict[int, list]] = {}

    def obter(self, area: str, acertos: int) -> Dict[str, float]:
        """
        Obtém valores TRI para uma área e número de acertos.

        Args:
            area: 'CH', 'CN', 'LC' ou 'MT'
            acertos: Número de acertos (0-45)

        Returns:
            Dict com 'tri_min', 'tri_med', 'tri_max'

        Raises:
            ValueError: Se a área não existe na tabela
        """
        if area not in self.lookup:
            raise ValueError(f"Área inválida: {area}. Disponíveis: {self.areas}")

        if acertos not in self.lookup[area]:
            # Se acertos está fora do range, usar valor máximo disponível
            max_acertos = max(self.lookup[area].keys())
            if acertos > max_acertos:
                acertos = max_acertos
            elif acertos < 0:
                acertos = 0

        return self.lookup[area][acertos]

    def obter_quantis(self, area: str, acertos: int) -> Optional[list]:
        """
        Percentis da nota para uma área e número de acertos, quando o ano tem
        distribuição completa (mín, p01, p05, p10, p25, p50, p75, p90, p95,
        p99, máx). Retorna None para anos que só têm mín/med/máx.
        """
        por_acertos = getattr(self, "quantis", {}).get(area)
        if not por_acertos:
            return None
        acertos = max(min(por_acertos), min(int(acertos), max(por_acertos)))
        return por_acertos.get(acertos)

    def validar(self) -> bool:
        """Valida integridade da tabela (monotonia da mediana)."""
        for area in self.lookup:
            assert 0 in self.lookup[area], f"Falta 0 acertos para {area}"
            tri_meds = [
                self.lookup[area][a]['tri_med']
                for a in sorted(self.lookup[area].keys())
            ]
            assert tri_meds == sorted(tri_meds), \
                f"TRI mediana não está monotônica para {area}"
        return True

    def info(self) -> Dict:
        """Retorna informações sobre a tabela carregada."""
        result = {
            'areas': self.areas,
            'acertos_por_area': {
                area: len(vals) for area, vals in self.lookup.items()
            },
            'limites_oficiais': {
                area: {
                    'min': TRI_MINIMA_OFICIAL.get(area),
                    'max': TRI_MAXIMA_OFICIAL.get(area),
                }
                for area in self.areas
            },
        }
        if self.ano is not None:
            result['ano'] = self.ano
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Multi-year support
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def carregar_multiano(
        cls, json_path: Optional[str] = None
    ) -> Dict[int, "TabelaReferenciaTRI"]:
        """
        Load multi-year reference JSON and return a dict of year -> instance.

        Args:
            json_path: Path to tri_tabela_referencia_multiano.json.
                       If None, uses default path inside app/referencias/.
        """
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'referencias',
                'tri_tabela_referencia_multiano.json',
            )

        with open(json_path, 'r') as f:
            raw = json.load(f)

        tabelas: Dict[int, TabelaReferenciaTRI] = {}
        for ano_str, areas_data in raw.items():
            instance = cls.__new__(cls)
            instance.ano = int(ano_str)
            instance.lookup = {}
            for area, acertos_dict in areas_data.items():
                instance.lookup[area] = {}
                for acertos_str, valores in acertos_dict.items():
                    acertos = int(acertos_str)
                    instance.lookup[area][acertos] = {
                        'tri_min': round(valores['tri_min'], 1),
                        'tri_med': round(valores['tri_med'], 1),
                        'tri_max': round(valores['tri_max'], 1),
                    }
            instance.areas = list(instance.lookup.keys())
            instance.quantis = {}
            tabelas[instance.ano] = instance

        cls._anexar_quantis(tabelas, os.path.dirname(json_path))
        return tabelas

    @staticmethod
    def _anexar_quantis(tabelas: Dict[int, "TabelaReferenciaTRI"], pasta: str) -> None:
        """Anexa percentis por acertos aos anos presentes em `tri_quantis_referencia.json`."""
        caminho = os.path.join(pasta, 'tri_quantis_referencia.json')
        if not os.path.isfile(caminho):
            return
        with open(caminho, 'r') as f:
            raw = json.load(f)
        for ano_str, areas_data in raw.items():
            tabela = tabelas.get(int(ano_str))
            if tabela is None:
                continue
            tabela.quantis = {
                area: {int(k): [float(v) for v in valores] for k, valores in por_acertos.items()}
                for area, por_acertos in areas_data.items()
            }
