"""
Gerenciador da tabela de referência TRI oficial do ENEM.

Carrega os dados de referência (tri_min, tri_med, tri_max) por área e número
de acertos, usados para ancoragem das notas em ambos os motores.
"""

import json
import os
from typing import Dict, Optional

# Limites máximos EMPIRICOS observados nos microdados ENEM 2022/2023/2024
# (max real + ~5 pts buffer para variabilidade de amostragem entre anos)
#
# Valores reais máximos observados (100K-200K alunos por ano):
#   LC: 2022=764.6  2023=758.6  2024=795.8  → max global 795.8 → teto 800
#   CH: 2022=829.1  2023=823.0  2024=819.7  → max global 829.1 → teto 835
#   CN: 2022=868.7  2023=868.4  2024=867.2  → max global 868.7 → teto 875
#   MT: 2022=983.9  2023=958.6  2024=961.9  → max global 983.9 → teto 990
#
# Importante: valores anteriores eram superestimados (LC 835 vs real 795).
# 99% dos alunos LC ficam abaixo de 670. O P99.99 (top 0.01%) é ~745.
# Apenas 1 em cada 10000 alunos chega perto do teto.
TRI_MAXIMA_OFICIAL = {
    'LC': 800.0,
    'CH': 835.0,
    'CN': 875.0,
    'MT': 990.0,
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
            tabelas[instance.ano] = instance

        return tabelas
