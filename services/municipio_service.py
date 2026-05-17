import geopandas as gpd
import json
from pathlib import Path
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from branca.element import Template, MacroElement
import rasterio
from folium.plugins import Draw, MousePosition
import folium
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from utils.utils import load_config


CAMADAS_EXTRAS_CIDADE_PADRAO = {
    "jaqueira": [
        {
            "nome": "Coleta Jaqueira",
            "caminho": "../COLETA_JAQUEIRA/COLETA_JAQUEIRA_PE.shp",
            "cor": "#ff7f11",
            "peso": 2,
            "opacidade": 0.85,
            "preenchimento": 0.2,
        }
    ]
}


def _obter_camadas_extras_cidade(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return CAMADAS_EXTRAS_CIDADE_PADRAO

    dados_cfg = config.get("dados") if isinstance(config.get("dados"), dict) else {}
    camadas_cfg = dados_cfg.get("camadas_extras_cidade") if isinstance(dados_cfg, dict) else None
    if not isinstance(camadas_cfg, dict):
        return CAMADAS_EXTRAS_CIDADE_PADRAO

    return camadas_cfg


def _resolver_caminho_camada(caminho: str) -> Path:
    p = Path(str(caminho)).expanduser()
    if p.is_absolute():
        return p

    app_root = Path(__file__).resolve().parents[1]
    candidatos = [
        (app_root / p).resolve(),
        (app_root.parent / p).resolve(),
    ]

    for c in candidatos:
        if c.exists():
            return c

    return candidatos[0]


def _preparar_geometrias_manchas(gdf: gpd.GeoDataFrame, camada_cfg: dict) -> gpd.GeoDataFrame:
    """Garante que a camada extra seja renderizada como mancha (polígono)."""

    if gdf.empty:
        return gdf

    geom_types = {str(t) for t in gdf.geometry.geom_type.dropna().unique()}
    tipos_ponto = {"Point", "MultiPoint"}
    tipos_linha = {"LineString", "MultiLineString"}

    # Se já for polígono, mantém como está.
    if geom_types & {"Polygon", "MultiPolygon"}:
        return gdf

    # Para pontos/linhas, cria buffer em metros para virar "mancha".
    if geom_types.issubset(tipos_ponto | tipos_linha):
        raio_m = float(camada_cfg.get("raio_manchas_m", 80.0))
        if raio_m <= 0:
            raio_m = 80.0

        gdf_m = gdf.to_crs(epsg=31985)
        gdf_m = gdf_m.copy()
        gdf_m["geometry"] = gdf_m.geometry.buffer(raio_m)
        gdf = gdf_m.to_crs(epsg=4326)

    return gdf


class MunicipioService:
    def __init__(self, municipios_path):
        self.municipios = gpd.read_file(municipios_path)

    def get_nome_cidades(self):
        lista_cidades = sorted(self.municipios["NM_MUN"].unique())

        return lista_cidades
    
    def get_gdf_municipio(self, nome_cidade):
        dados_municipios = self.municipios.loc[self.municipios['NM_MUN'] == nome_cidade]
        gdf = gpd.GeoDataFrame(dados_municipios, geometry=dados_municipios['geometry'])

        return gdf
    
    def gerar_mapa_base(self):
        mapa = folium.Map(
            location=[-8.38, -37.86],
            zoom_start=7,
            tiles=None
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satélite",
            overlay=False,
            control=False
        ).add_to(mapa)

        return mapa._repr_html_()
    
    def gerar_mapa_municipio(self, nome_cidade, raster_path):
        dados_municipios = self.municipios.loc[self.municipios['NM_MUN'] == nome_cidade]
        gdf = gpd.GeoDataFrame(dados_municipios, geometry=dados_municipios['geometry'])

        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        centro = gdf.geometry.centroid.iloc[0]
        lat, lon = centro.y, centro.x

        # Base map
        # Importante: não passar a URL de tiles direto no folium.Map, senão o LayerControl
        # pode exibir a própria URL como "nome" da camada base.
        m = folium.Map(
            location=[lat, lon],
            zoom_start=11,
            tiles=None,
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satélite",
            overlay=False,
            control=False,
        ).add_to(m)

        # GeoJson do município
        folium.GeoJson(
            gdf,
            name="Limite do Município",
            control=False, # Não mostrar no LayerControl
            style_function=lambda x: {
                'fillColor': 'yellow',
                'color': 'green',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)

        # Camadas extras por cidade (configuráveis em static/config/config.json).
        config = load_config() or {}
        camadas_por_cidade = _obter_camadas_extras_cidade(config)
        cidade_key = str(nome_cidade).strip().lower()
        for camada in camadas_por_cidade.get(cidade_key, []):
            caminho = camada.get("caminho") if isinstance(camada, dict) else None
            if not isinstance(caminho, str) or not caminho.strip():
                continue

            shp_path = _resolver_caminho_camada(caminho)
            if not shp_path.exists():
                print(f"Camada extra não encontrada: {shp_path}")
                continue

            try:
                gdf_extra = gpd.read_file(shp_path)
                if gdf_extra.empty:
                    continue
                if gdf_extra.crs is None:
                    gdf_extra = gdf_extra.set_crs(epsg=4326)
                elif str(gdf_extra.crs).upper() != "EPSG:4326":
                    gdf_extra = gdf_extra.to_crs(epsg=4326)

                gdf_extra = _preparar_geometrias_manchas(gdf_extra, camada)

                folium.GeoJson(
                    gdf_extra,
                    name=str(camada.get("nome") or shp_path.stem),
                    style_function=lambda _x, _camada=camada: {
                        "color": str(_camada.get("cor", "#ff7f11")),
                        "weight": float(_camada.get("peso", 2)),
                        "opacity": float(_camada.get("opacidade", 0.85)),
                        "fillColor": str(_camada.get("cor", "#ff7f11")),
                        "fillOpacity": float(_camada.get("preenchimento", 0.2)),
                    },
                ).add_to(m)
            except Exception as e:
                print(f"Erro ao carregar camada extra '{shp_path}': {e}")

        # Raster overlay
        with rasterio.open(raster_path) as src:
            if src.nodata is None:
                dtype = src.dtypes[0]
                if np.issubdtype(dtype, np.integer):
                    nodata = 0
                else:
                    nodata = np.nan
            else:
                nodata = src.nodata
            
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': 'EPSG:4326',
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': nodata
            })

            # Inicializar com NoData evita "buracos" por pixels não escritos pelo reproject.
            # (especialmente perceptível em bordas perto de água / recortes)
            data_reproj = np.full((height, width), nodata, dtype=src.meta['dtype'])
            rp_kwargs = {}
            if not (isinstance(nodata, float) and np.isnan(nodata)):
                rp_kwargs = {"src_nodata": nodata, "dst_nodata": nodata}

            reproject(
                source=rasterio.band(src, 1),
                destination=data_reproj,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
                **rp_kwargs,
            )
            # Normalizar valores inválidos para nodata antes de mascarar (evita NaN virar transparência)
            if isinstance(data_reproj, np.ndarray) and np.issubdtype(data_reproj.dtype, np.floating):
                data_reproj = data_reproj.astype(np.float32, copy=False)
                bad = ~np.isfinite(data_reproj)
                if np.any(bad):
                    data_reproj[bad] = nodata if not (isinstance(nodata, float) and np.isnan(nodata)) else -9999.0

            # Máscara: NoData e valores inválidos
            data = np.ma.masked_invalid(data_reproj)
            if not (isinstance(nodata, float) and np.isnan(nodata)):
                data = np.ma.masked_equal(data, nodata)
            bounds = rasterio.transform.array_bounds(height, width, transform)

        # Overlay de corpos d'água (azul claro), para distinguir das áreas de risco
        # Usa os IDs configurados em config.json e reprojeta para o MESMO grid EPSG:4326 do overlay de risco.
        cfg_water = load_config() or {}
        uso_cfg = (cfg_water.get("criterios") or {}).get("uso_do_solo") or {}
        classes_cfg = uso_cfg.get("classes") or {}
        water_ids = (((classes_cfg.get("corpos_dagua") or {}).get("ids")) if isinstance(classes_cfg, dict) else None)
        if not isinstance(water_ids, list) or not all(isinstance(x, (int, float)) for x in water_ids):
            water_ids = [26, 31, 33]
        water_ids = [int(x) for x in water_ids]

        # Camada extra: uso e ocupação do solo (MapBiomas) com cores (paleta padrão; pode ser ajustada no config)
        def _hex_to_rgb(h: str) -> tuple[int, int, int]:
            s = (h or "").strip().lstrip("#")
            if len(s) != 6:
                return (176, 176, 176)
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

        try:
            uso_solo_path = "dados/uso-do-solo-pernambuco-2023.tif"
            with rasterio.open(uso_solo_path) as uso_src:
                gdf_uso = gdf
                if uso_src.crs is not None and str(uso_src.crs).upper() != "EPSG:4326":
                    gdf_uso = gdf.to_crs(uso_src.crs)

                out_image, out_transform = mask(
                    uso_src,
                    gdf_uso.geometry,
                    crop=True,
                    filled=True,
                    nodata=(uso_src.nodata if uso_src.nodata is not None else 0),
                )

                uso_crop = out_image[0]
                if uso_src.nodata is not None:
                    valid_uso = uso_crop != uso_src.nodata
                else:
                    valid_uso = np.ones_like(uso_crop, dtype=bool)

                # Reprojetar o raster categórico de uso do solo para o MESMO grid do overlay de risco (EPSG:4326)
                uso_dst = np.zeros((height, width), dtype=np.int32)
                reproject(
                    source=uso_crop.astype(np.int32, copy=False),
                    destination=uso_dst,
                    src_transform=out_transform,
                    src_crs=uso_src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                    src_nodata=(int(uso_src.nodata) if uso_src.nodata is not None else 0),
                    dst_nodata=0,
                )

                # Paleta: preferir a colortable do próprio GeoTIFF (cores oficiais do MapBiomas).
                # Se não existir, cai no fallback (cores por grupo) para ainda permitir depuração.
                id_to_rgb: dict[int, tuple[int, int, int]] = {}
                try:
                    cmap = uso_src.colormap(1)  # dict: value -> (r,g,b[,a])
                except Exception:
                    cmap = None

                if isinstance(cmap, dict) and cmap:
                    for k, rgba in cmap.items():
                        if rgba is None:
                            continue
                        if isinstance(rgba, (list, tuple)) and len(rgba) >= 3:
                            id_to_rgb[int(k)] = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
                else:
                    palette_cfg = ((cfg_water.get("mapbiomas") or {}).get("palette")) if isinstance(cfg_water, dict) else None
                    if not isinstance(palette_cfg, dict):
                        palette_cfg = {}

                    veget_ids = [int(x) for x in (classes_cfg.get("vegetacao", {}) or {}).get("ids", [])] if isinstance(classes_cfg, dict) else []
                    plant_ids = [int(x) for x in (classes_cfg.get("regeneracao_floresta_plantada", {}) or {}).get("ids", [])] if isinstance(classes_cfg, dict) else []
                    agri_ids = [int(x) for x in (classes_cfg.get("agricultura", {}) or {}).get("ids", [])] if isinstance(classes_cfg, dict) else []
                    urb_ids = [int(x) for x in (classes_cfg.get("urbano", {}) or {}).get("ids", [])] if isinstance(classes_cfg, dict) else []
                    nao_obs_ids = [int(x) for x in (classes_cfg.get("nao_observado", {}) or {}).get("ids", [])] if isinstance(classes_cfg, dict) else []

                    default_groups = {
                        "vegetacao": ("#1B5E20", veget_ids),
                        "floresta_plantada": ("#6A1B9A", plant_ids),
                        "agricultura": ("#F9A825", agri_ids),
                        "urbano": ("#C62828", urb_ids),
                        "corpos_dagua": ("#1E88E5", water_ids),
                        "nao_observado": ("#9E9E9E", nao_obs_ids),
                    }

                    for group_name, (default_hex, ids_list) in default_groups.items():
                        override_hex = palette_cfg.get(group_name) if isinstance(palette_cfg, dict) else None
                        rgb = _hex_to_rgb(override_hex) if isinstance(override_hex, str) else _hex_to_rgb(default_hex)
                        for _id in ids_list:
                            id_to_rgb[int(_id)] = rgb

                uso_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                # Base: pixels válidos que não caírem em nenhuma cor ficam em cinza (ajuda a achar IDs não previstos)
                unknown_rgb = _hex_to_rgb("#B0B0B0")
                unknown_mask = (uso_dst != 0)
                uso_rgba[unknown_mask, 0] = unknown_rgb[0]
                uso_rgba[unknown_mask, 1] = unknown_rgb[1]
                uso_rgba[unknown_mask, 2] = unknown_rgb[2]
                uso_rgba[unknown_mask, 3] = 255

                for _id, (r, g, b) in id_to_rgb.items():
                    m_id = (uso_dst == _id)
                    if not np.any(m_id):
                        continue
                    uso_rgba[m_id, 0] = r
                    uso_rgba[m_id, 1] = g
                    uso_rgba[m_id, 2] = b
                    uso_rgba[m_id, 3] = 255

                # Pixels 0 (NoData) ficam transparentes
                uso_rgba[uso_dst == 0, 3] = 0

                # Não mostrar por padrão (fica disponível no LayerControl)
                uso_group = folium.FeatureGroup(
                    name="Uso e ocupação do solo (MapBiomas)",
                    show=False,
                    overlay=True,
                    control=True,
                )
                folium.raster_layers.ImageOverlay(
                    name="Uso e ocupação do solo (MapBiomas)",
                    image=uso_rgba,
                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                    opacity=0.85,
                    interactive=False,
                    cross_origin=False,
                ).add_to(uso_group)
                uso_group.add_to(m)

                water_src = (np.isin(uso_crop, water_ids) & valid_uso).astype(np.uint8)

                water_dst = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=water_src,
                    destination=water_dst,
                    src_transform=out_transform,
                    src_crs=uso_src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=0,
                )

            water_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            water_mask = water_dst == 1
            # azul claro
            water_rgba[water_mask, 0] = 120  # R
            water_rgba[water_mask, 1] = 200  # G
            water_rgba[water_mask, 2] = 255  # B
            water_rgba[water_mask, 3] = 255  # A (opacity controlado pelo Leaflet)

            water_group = folium.FeatureGroup(
                name="Corpos d'água",
                show=True,
                overlay=True,
                control=True,
            )
            folium.raster_layers.ImageOverlay(
                name="Corpos d'água",
                image=water_rgba,
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.55,
                interactive=False,
                cross_origin=False,
            ).add_to(water_group)
            water_group.add_to(m)
        except Exception as e:
            # Se faltar o raster de uso do solo ou der erro de reprojeção, apenas não desenha o overlay.
            print(f"Erro ao carregar camadas de uso/água: {e}")
            pass

        # Camada de mancha de inundação (HEC-RAS)
        # mancha_group = folium.FeatureGroup(
        #     name="Mancha de inundação (HEC-RAS)",
        #     show=False,
        #     overlay=True,
        #     control=True,
        # )
        # try:
        #     mancha_path = "dados/manchas-inundacao-normalizado.tif"
        #     if Path(mancha_path).exists():
        #         with rasterio.open(mancha_path) as mancha_src:
        #             # Ler dados da mancha
        #             mancha_data = mancha_src.read(1).astype(np.float32)
        #             mancha_nodata = mancha_src.nodata
        #
        #             # Calcular transformação para EPSG:4326 com as mesmas dimensões do grid de risco
        #             mancha_transform, mancha_w, mancha_h = calculate_default_transform(
        #                 mancha_src.crs, "EPSG:4326", mancha_src.width, mancha_src.height, *mancha_src.bounds
        #             )
        #
        #             # Reprojetar mancha para o mesmo grid do overlay de risco (EPSG:4326)
        #             # Importante: manter float para não truncar rasters normalizados (0..1) ao converter para uint8
        #             mancha_reproj = np.full((height, width), 0.0, dtype=np.float32)
        #             reproject(
        #                 source=mancha_data.astype(np.float32),
        #                 destination=mancha_reproj,
        #                 src_transform=mancha_src.transform,
        #                 src_crs=mancha_src.crs,
        #                 dst_transform=transform,
        #                 dst_crs="EPSG:4326",
        #                 resampling=Resampling.nearest,
        #                 src_nodata=float(mancha_nodata)
        #                 if (mancha_nodata is not None and np.isfinite(mancha_nodata))
        #                 else 0.0,
        #                 dst_nodata=0.0,
        #             )
        #
        #             # Converter para RGBA: azul semi-transparente onde houver inundação
        #             mancha_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        #             # Onde mancha_reproj > 0 (inundação), colorir em azul
        #             # (limiar pequeno para evitar problemas numéricos/ruído)
        #             flood_mask = mancha_reproj > 1e-6
        #             mancha_rgba[flood_mask, 0] = 0      # R
        #             mancha_rgba[flood_mask, 1] = 100    # G
        #             mancha_rgba[flood_mask, 2] = 200    # B
        #             # Alpha total no pixel; a transparência final fica controlada pelo parâmetro opacity do overlay
        #             mancha_rgba[flood_mask, 3] = 255
        #
        #             folium.raster_layers.ImageOverlay(
        #                 name="Mancha de inundação (HEC-RAS)",
        #                 image=mancha_rgba,
        #                 bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        #                 opacity=0.8,
        #                 interactive=False,
        #                 cross_origin=False,
        #             ).add_to(mancha_group)
        # except Exception as e:
        #     # Se faltar a mancha ou der erro, apenas não desenha
        #     print(f"Erro ao carregar camada HEC-RAS: {e}")
        #     import traceback
        #     traceback.print_exc()
        #
        # # Sempre adicionar o grupo ao mapa, mesmo se vazio, para aparecer no LayerControl
        # mancha_group.add_to(m)

        # Converter para imagem normalizada (0-255) para overlay
        # Verificar se deve usar classes discretas (4 classes) ou contínuo
        config = load_config() or {}
        tipo_risco = ((config.get("visualizacao") or {}).get("tipo_risco") or "continuo").lower()
        usar_classes_4 = tipo_risco == "classes_4"
        
        if usar_classes_4:
            # Reclassificar em 4 classes discretas baseadas no intervalo [vmin, vmax]
            vmin = data.min()
            vmax = data.max()
            if vmin < vmax:
                intervalo = (vmax - vmin) / 4.0
                norm_data = np.zeros_like(data, dtype=np.float32)
                
                mask1 = (data >= vmin) & (data < vmin + intervalo)
                mask2 = (data >= vmin + intervalo) & (data < vmin + 2*intervalo)
                mask3 = (data >= vmin + 2*intervalo) & (data < vmin + 3*intervalo)
                mask4 = (data >= vmin + 3*intervalo)
                
                norm_data[mask1] = 0.125  # Classe 1 (menor risco)
                norm_data[mask2] = 0.375  # Classe 2
                norm_data[mask3] = 0.625  # Classe 3
                norm_data[mask4] = 0.875  # Classe 4 (maior risco)
            else:
                norm_data = np.zeros_like(data, dtype=np.float32)
        else:
            # Modo contínuo: normalização linear
            norm_data = (data - data.min()) / (data.max() - data.min())
        
        rgba = plt.cm.RdYlGn_r(norm_data)  # colormap matplotlib
        rgba = (rgba[:, :, :4] * 255).astype(np.uint8)  # converter para 0-255

        # Adicionar ao mapa dentro de um FeatureGroup para que apareça no LayerControl
        risco_group = folium.FeatureGroup(
            name="Zonas de risco alagamento",
            show=True,
            overlay=True,
            control=True,
        )
        risco_overlay = folium.raster_layers.ImageOverlay(
            name="Zonas de risco alagamento",
            image=rgba,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.6,
            interactive=False,
            cross_origin=False
        ).add_to(risco_group)
        risco_group.add_to(m)
        risco_layer_js = risco_group.get_name()

        # Importante: o JS de atualização usa setUrl(), que existe no ImageOverlay (não no FeatureGroup)
        overlay = risco_overlay

        # Plugins

        # Popup customizado: risco + uso do solo no ponto clicado
        chamados_legend_html = None
        chamados_layer_js = None
        map_name = m.get_name()
        overlay_name = overlay.get_name()
        cidade_js = str(nome_cidade).replace("\\", "\\\\").replace("\"", "\\\"")

        # Pesos iniciais: usar os mesmos pesos do AHP (pairwise) do config
        cfg = load_config() or {}
        p = cfg.get("pesos") if isinstance(cfg.get("pesos"), dict) else {}

        uso_vs_declividade = float(p.get("uso_vs_declividade", 1 / 5))
        uso_vs_fluxo = float(p.get("uso_vs_fluxo", 3))
        uso_vs_hipsometria = float(p.get("uso_vs_hipsometria", 1 / 5))
        declividade_vs_fluxo = float(p.get("declividade_vs_fluxo", 3))
        declividade_vs_hipsometria = float(p.get("declividade_vs_hipsometria", 1))
        fluxo_vs_hipsometria = float(p.get("fluxo_vs_hipsometria", 1 / 5))

        pairwise = np.array(
            [
                uso_vs_declividade,
                uso_vs_fluxo,
                uso_vs_hipsometria,
                declividade_vs_fluxo,
                declividade_vs_hipsometria,
                fluxo_vs_hipsometria,
            ],
            dtype=float,
        )
        if (not np.isfinite(pairwise).all()) or np.any(pairwise <= 0):
            uso_vs_declividade = 1 / 5
            uso_vs_fluxo = 3
            uso_vs_hipsometria = 1 / 5
            declividade_vs_fluxo = 3
            declividade_vs_hipsometria = 1
            fluxo_vs_hipsometria = 1 / 5

        A = np.array(
            [
                [1, uso_vs_declividade, uso_vs_fluxo, uso_vs_hipsometria],
                [1 / uso_vs_declividade, 1, declividade_vs_fluxo, declividade_vs_hipsometria],
                [1 / uso_vs_fluxo, 1 / declividade_vs_fluxo, 1, fluxo_vs_hipsometria],
                [1 / uso_vs_hipsometria, 1 / declividade_vs_hipsometria, 1 / fluxo_vs_hipsometria, 1],
            ],
            dtype=float,
        )
        vals, vecs = np.linalg.eig(A)
        idx = np.argmax(vals.real)
        w_init = vecs[:, idx].real
        w_init = w_init / w_init.sum()
        w_uso0, w_decl0, w_flux0, w_hipso0 = [float(x) for x in w_init]

        # Layout final: sliders sempre "floating" dentro do mapa (Folium iframe)
        sliders_layout = "floating"

        sliders_inner_html = f"""
            <div style="
                width: 100%;
                font-size: 13px;
                background-color: white;
                border: 2px solid grey;
                border-radius: 6px;
                padding: 10px;
                box-shadow: 3px 3px 5px rgba(0,0,0,0.25);
            ">
                <b>Pesos (AHP simplificado)</b><br>
                <div style="margin-top:6px; display:flex; gap:8px; align-items:center;">
                    <button id="w_reset_btn" type="button" style="
                        padding: 4px 8px;
                        border: 1px solid #777;
                        background: #f7f7f7;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Resetar</button>
                    <span style="color:#666; font-size:12px;">volta ao padrão do AHP</span>
                </div>
                <div style=\"margin-top:6px;\">
                    <label>Uso do solo: <span id=\"w_uso_val\">{w_uso0:.4g}</span></label>
                    <div style=\"display:flex; gap:6px; align-items:center;\">
                        <input id=\"w_uso\" type=\"range\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_uso0:.6f}\" style=\"flex:1;\" />
                        <input id=\"w_uso_num\" type=\"number\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_uso0:.6f}\" style=\"width:82px;\" />
                    </div>
                </div>
                <div>
                    <label>Declividade: <span id=\"w_decl_val\">{w_decl0:.4g}</span></label>
                    <div style=\"display:flex; gap:6px; align-items:center;\">
                        <input id=\"w_decl\" type=\"range\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_decl0:.6f}\" style=\"flex:1;\" />
                        <input id=\"w_decl_num\" type=\"number\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_decl0:.6f}\" style=\"width:82px;\" />
                    </div>
                </div>
                <div>
                    <label>Fluxo: <span id=\"w_flux_val\">{w_flux0:.4g}</span></label>
                    <div style=\"display:flex; gap:6px; align-items:center;\">
                        <input id=\"w_flux\" type=\"range\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_flux0:.6f}\" style=\"flex:1;\" />
                        <input id=\"w_flux_num\" type=\"number\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_flux0:.6f}\" style=\"width:82px;\" />
                    </div>
                </div>
                <div>
                    <label>Hipsometria: <span id=\"w_hipso_val\">{w_hipso0:.4g}</span></label>
                    <div style=\"display:flex; gap:6px; align-items:center;\">
                        <input id=\"w_hipso\" type=\"range\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_hipso0:.6f}\" style=\"flex:1;\" />
                        <input id=\"w_hipso_num\" type=\"number\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"{w_hipso0:.6f}\" style=\"width:82px;\" />
                    </div>
                </div>
                <div style=\"margin-top:8px; color:#444;\">
                    <span id=\"w_status\">Arraste os sliders para atualizar o mapa</span>
                </div>
            </div>
        """

        sliders_template = f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 260px;
            z-index: 9999;
            font-size: 13px;
            background-color: white;
            border: 2px solid grey;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 3px 3px 5px rgba(0,0,0,0.25);
        ">
            <button type="button" onclick="
                var el = document.getElementById('ahp_panel');
                if (el) el.style.display = (el.style.display === 'none') ? 'block' : 'none';
            " style="
                width: 100%;
                padding: 6px 8px;
                border: 1px solid #777;
                background: #f7f7f7;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            ">Ajustes AHP</button>
            <div id="ahp_panel" style="margin-top:8px; display:none;">
                {sliders_inner_html}
            </div>
        </div>
        {{% endmacro %}}
        """

        sliders_macro = MacroElement()
        sliders_macro._template = Template(sliders_template)
        m.add_child(sliders_macro)

        click_template = f"""
        {{% macro script(this, kwargs) %}}
        function _normW(raw) {{
            if (!raw || raw.length !== 4) return [0.25, 0.25, 0.25, 0.25];
            const clean = raw.map(v => (isFinite(v) && v >= 0) ? Number(v) : 0);
            let sum = clean.reduce((a,b) => a + b, 0);
            if (!isFinite(sum) || sum <= 0) sum = 1;
            return clean.map(v => v / sum);
        }}

        // Pesos atuais (para o clique/popup e para sincronizar overlay)
        window.__currentWeights = _normW([{w_uso0}, {w_decl0}, {w_flux0}, {w_hipso0}]);
        // Pesos padrão (iniciais) para o botão de reset
        window.__initialWeights = [{w_uso0}, {w_decl0}, {w_flux0}, {w_hipso0}];

        function _fmt(v) {{
            if (v === null || v === undefined) return '—';
            if (typeof v === 'number') return (Math.round(v * 100) / 100).toString();
            return v.toString();
        }}

        function _fmtSig(v) {{
            if (!isFinite(v)) return '—';
            // 4 algarismos significativos
            const s = Number(v).toPrecision(4);
            // Evitar notação científica quando dá para mostrar como decimal curto
            const n = Number(s);
            if (isFinite(n) && Math.abs(n) >= 0.001 && Math.abs(n) < 1000) return n.toString();
            return s;
        }}

        function _getEl(id) {{ return document.getElementById(id); }}

        function _readPair(s) {{
            const num = _getEl(`w_${{s}}_num`);
            const rng = _getEl(`w_${{s}}`);
            const vNum = num ? parseFloat(num.value) : NaN;
            const vRng = rng ? parseFloat(rng.value) : NaN;
            return {{ num, rng, vNum, vRng }};
        }}

        function _readRawWeights(sourceId) {{
            const ids = ['uso','decl','flux','hipso'];
            const raw = ids.map(s => {{
                const {{ num, rng, vNum, vRng }} = _readPair(s);

                // Se o evento veio do range daquele peso...
                if (sourceId === `w_${{s}}` && isFinite(vRng)) return vRng;
                // Se o evento veio do numérico daquele peso...
                if (sourceId === `w_${{s}}_num` && isFinite(vNum)) return vNum;

                // Caso geral: preferir o elemento que está em foco
                if (document.activeElement === rng && isFinite(vRng)) return vRng;
                if (document.activeElement === num && isFinite(vNum)) return vNum;

                // Fallback: preferir num se válido; senão range
                if (isFinite(vNum)) return vNum;
                if (isFinite(vRng)) return vRng;
                return 0;
            }});
            return raw;
        }}

        function _syncInputs(raw, sourceId) {{
            const ids = ['uso','decl','flux','hipso'];
            ids.forEach((s, i) => {{
                const v = raw[i];
                const num = _getEl(`w_${{s}}_num`);
                const rng = _getEl(`w_${{s}}`);

                // Atualiza o campo oposto ao que disparou o evento
                if (num && sourceId !== `w_${{s}}_num` && document.activeElement !== num) num.value = v.toFixed(6);
                if (rng && sourceId !== `w_${{s}}` && document.activeElement !== rng) rng.value = v.toFixed(6);
            }});
        }}

        function _getWeights() {{
            const raw = _readRawWeights();
            let sum = raw.reduce((a,b) => a + b, 0);
            if (!isFinite(sum) || sum <= 0) sum = 1;
            const w = raw.map(v => v / sum);
            window.__currentWeights = w;
            return w;
        }}

        function _renderWeights(w) {{
            const el0 = document.getElementById('w_uso_val');
            const el1 = document.getElementById('w_decl_val');
            const el2 = document.getElementById('w_flux_val');
            const el3 = document.getElementById('w_hipso_val');
            if (el0) el0.textContent = _fmtSig(w[0]);
            if (el1) el1.textContent = _fmtSig(w[1]);
            if (el2) el2.textContent = _fmtSig(w[2]);
            if (el3) el3.textContent = _fmtSig(w[3]);
        }}

        let _timer = null;
        function _scheduleOverlayUpdate(sourceId) {{
            const statusEl = document.getElementById('w_status');
            const raw = _readRawWeights(sourceId);
            _syncInputs(raw, sourceId);
            let sum = raw.reduce((a,b) => a + b, 0);
            if (!isFinite(sum) || sum <= 0) sum = 1;
            const w = raw.map(v => v / sum);
            _renderWeights(w);
            if (_timer) clearTimeout(_timer);
            _timer = setTimeout(() => {{
                if (statusEl) statusEl.textContent = 'Atualizando...';
                const url = `/overlay_risco?cidade=${{encodeURIComponent(\"{cidade_js}\")}}` +
                            `&w_uso=${{w[0]}}&w_decl=${{w[1]}}&w_flux=${{w[2]}}&w_hipso=${{w[3]}}`;
                fetch(url)
                    .then(r => r.json())
                    .then(data => {{
                        if (data.status !== 'ok') {{
                            if (statusEl) statusEl.textContent = data.mensagem || 'Falha ao atualizar.';
                            return;
                        }}
                        {overlay_name}.setUrl(data.url);
                        if (statusEl) statusEl.textContent = 'Atualizado';
                    }})
                    .catch(err => {{
                        if (statusEl) statusEl.textContent = 'Erro: ' + err;
                    }});
            }}, 250);
        }}

        function _setWeightsRaw(raw) {{
            const ids = ['uso','decl','flux','hipso'];
            ids.forEach((s, i) => {{
                const v = (raw && raw.length === 4 && isFinite(raw[i])) ? Number(raw[i]) : 0;
                const num = document.getElementById(`w_${{s}}_num`);
                const rng = document.getElementById(`w_${{s}}`);
                if (num) num.value = v.toFixed(6);
                if (rng) rng.value = v.toFixed(6);
            }});
            _scheduleOverlayUpdate('reset');
        }}

        // Conecta sliders
        ['w_uso','w_decl','w_flux','w_hipso','w_uso_num','w_decl_num','w_flux_num','w_hipso_num'].forEach(id => {{
            const el = document.getElementById(id);
            if (el) el.addEventListener('input', () => _scheduleOverlayUpdate(id));
        }});

        // Botão de reset
        const resetBtn = document.getElementById('w_reset_btn');
        if (resetBtn) {{
            resetBtn.addEventListener('click', (ev) => {{
                ev.preventDefault();
                ev.stopPropagation();
                _setWeightsRaw(window.__initialWeights);
            }});
        }}
        // Render inicial com maior precisão
        _renderWeights(_getWeights());

        {map_name}.on('click', function(e) {{
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            const w = _getWeights();
            const url = `/valor_ponto?cidade=${{encodeURIComponent("{cidade_js}")}}&lat=${{lat}}&lon=${{lon}}` +
                        `&w_uso=${{w[0]}}&w_decl=${{w[1]}}&w_flux=${{w[2]}}&w_hipso=${{w[3]}}`;

            fetch(url)
                .then(r => r.json())
                .then(data => {{
                    let html = `<b>Coordenadas</b><br>Lat: ${{lat.toFixed(5)}}<br>Lon: ${{lon.toFixed(5)}}`;

                    if (data.status !== 'ok') {{
                        html += `<br><br><b>Erro</b><br>${{data.mensagem || 'Falha ao consultar o ponto'}}`;
                    }} else {{
                        if (data.bairro) {{
                            html += `<br><b>Bairro</b>: ${{data.bairro}}`;
                        }}
                        html += `<br><br><b>Risco</b>: ${{_fmt(data.risco)}}`;
                        const usoClasse = data.uso_classe ? data.uso_classe : '—';
                        html += `<br><b>Uso do solo</b>: ${{usoClasse}}`;
                        html += `<br><b>Uso ID</b>: ${{_fmt(data.uso_id)}}`;
                            html += `<br><b>Declividade (°)</b>: ${{_fmt(data.declividade)}}`;
                            html += `<br><b>Elevação (m)</b>: ${{_fmt(data.elevacao)}}`;
                            html += `<br><b>Fluxo acumulado</b>: ${{_fmt(data.fluxo_acumulado)}}`;
                    }}

                    L.popup().setLatLng(e.latlng).setContent(html).openOn({map_name});
                }})
                .catch(err => {{
                    const html = `<b>Coordenadas</b><br>Lat: ${{lat.toFixed(5)}}<br>Lon: ${{lon.toFixed(5)}}` +
                                 `<br><br><b>Erro</b><br>${{err}}`;
                    L.popup().setLatLng(e.latlng).setContent(html).openOn({map_name});
                }});
        }});
        {{% endmacro %}}
        """

        click_macro = MacroElement()
        click_macro._template = Template(click_template)
        m.add_child(click_macro)

        # Ferramenta de desenho (ponto e retângulo)
        Draw(
            draw_options={
                'polyline': False,
                'polygon': True,
                'circle': False,
                'rectangle': True,
                'marker': True
            },
            edit_options={'edit': True}
        ).add_to(m)

        template = """
        {% macro html(this, kwargs) %}
        <div id="legend_risco" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 150px;
            height: 160px;
            z-index:9999;
            font-size:14px;
            background-color: white;
            border:2px solid grey;
            border-radius:5px;
            padding: 10px;
            box-shadow: 3px 3px 5px rgba(0,0,0,0.4);
            display: none;
        ">
            <b>Risco de Alagamento</b><br>
            <i style="background:#d73027;width:20px;height:20px;display:inline-block;margin-right:5px;"></i> Alto<br>
            <i style="background:#ffffb2;width:20px;height:20px;display:inline-block;margin-right:5px;"></i> Moderado<br>
            <i style="background:#78c679;width:20px;height:20px;display:inline-block;margin-right:5px;"></i> Baixo<br>
            <i style="background:#006837;width:20px;height:20px;display:inline-block;margin-right:5px;"></i> Muito Baixo<br>
        </div>
        {% endmacro %}
        """

        macro = MacroElement()
        macro._template = Template(template)

        # Marcadores de pontos de alagamento: apenas para Recife
        if str(nome_cidade).strip().casefold() == "recife":
            pontos_alagamento = [
                ("Rua Imperial, bairro de São José", -8.07581, -34.89415),
                ("Rua Nicolau Pereira", -8.07804, -34.90558),
                ("Av. Eng. Abdias de Carvalho", -8.06123, -34.92227),
                ("Av. Dois Rios", -8.11289, -34.93864),
                ("Av. Mal Mascarenhas de Moraes", -8.11383, -34.91281),
                ("Av. Recife próximo ao cruzamento com a Rua João Cabral de Melo Neto", -8.07953, -34.93374),
                ("Av. Abdias de Carvalho, no cruzamento com a rua Delmiro Gouveia", -8.06252, -34.93219),
                ("Av. Norte Miguel Arraes de Alencar, ao lado do Senai", -8.04713, -34.87757),
            ]

            pontos_group = folium.FeatureGroup(
                name="Pontos de alagamento conhecidos",
                show=False,
                overlay=True,
                control=True,
            )

            for endereco, lat_ponto, lon_ponto in pontos_alagamento:
                folium.Marker(
                    location=[lat_ponto, lon_ponto],
                    popup=folium.Popup(endereco, max_width=300),
                    icon=folium.Icon(color="red", icon="info-sign"),
                ).add_to(pontos_group)

            pontos_group.add_to(m)

        chamados_legend_html = None

        # Fronteiras dos bairros (Recife) - adicionar por ultimo para ficar acima do raster
        if nome_cidade.strip().lower() == "recife":
            try:
                bairros_shp = Path("dados/Bairros-Recife/Bairros.shp")
                if bairros_shp.exists():
                    gdf_bairros = gpd.read_file(bairros_shp)
                    if gdf_bairros.crs is None:
                        # CRS conhecido do arquivo (SIRGAS 2000 / UTM 25S)
                        gdf_bairros = gdf_bairros.set_crs(epsg=31985)
                    if str(gdf_bairros.crs).upper() != "EPSG:4326":
                        gdf_bairros = gdf_bairros.to_crs(epsg=4326)

                    print(f"Bairros carregados: {len(gdf_bairros)}")

                    # Evitar erro de serializacao: converter campos datetime para string
                    for col, dtype in gdf_bairros.dtypes.items():
                        if "datetime" in str(dtype):
                            gdf_bairros[col] = gdf_bairros[col].astype(str)

                    bairros_calls_path = Path("outputs/bairros_alagamento_recife.json")
                    bairros_info = {}
                    if bairros_calls_path.exists():
                        try:
                            with bairros_calls_path.open("r", encoding="utf-8") as f:
                                bairros_data = json.load(f)
                            if isinstance(bairros_data, list):
                                for item in bairros_data:
                                    nome = str(item.get("bairro", "")).strip()
                                    if not nome:
                                        continue
                                    try:
                                        chamados_val = float(item.get("chamados"))
                                    except Exception:
                                        chamados_val = 0.0
                                    try:
                                        area_km2 = float(item.get("area_km2"))
                                    except Exception:
                                        area_km2 = np.nan
                                    try:
                                        pct_alto = float(item.get("pct_area_alto"))
                                    except Exception:
                                        pct_alto = np.nan

                                    bairros_info[nome.casefold()] = {
                                        "chamados": chamados_val,
                                        "area_km2": area_km2,
                                        "pct_area_alto": pct_alto,
                                    }
                        except Exception as e:
                            print(f"Erro ao carregar chamados por bairro: {e}")

                    name_col = None
                    for cand in ["bairro", "BAIRRO", "NOME", "nome", "NM_BAIRRO", "NM_BAIRR", "Bairro"]:
                        if cand in gdf_bairros.columns:
                            name_col = cand
                            break
                    if name_col is None:
                        for col in gdf_bairros.columns:
                            if col == "geometry":
                                continue
                            if gdf_bairros[col].dtype == object:
                                name_col = col
                                break

                    if name_col:
                        bairro_key = (
                            gdf_bairros[name_col]
                            .astype(str)
                            .str.strip()
                            .str.casefold()
                        )
                        info_map = bairro_key.map(bairros_info)
                        gdf_bairros["__chamados"] = (
                            info_map
                            .map(lambda v: v.get("chamados") if isinstance(v, dict) else 0.0)
                            .fillna(0.0)
                        )
                        gdf_bairros["__area_km2"] = info_map.map(
                            lambda v: v.get("area_km2") if isinstance(v, dict) else np.nan
                        )
                        gdf_bairros["__pct_area_alto"] = info_map.map(
                            lambda v: v.get("pct_area_alto") if isinstance(v, dict) else np.nan
                        )
                    else:
                        gdf_bairros["__chamados"] = 0.0
                        gdf_bairros["__area_km2"] = np.nan
                        gdf_bairros["__pct_area_alto"] = np.nan

                    chamados_series = gdf_bairros["__chamados"].astype(float)
                    max_real = float(chamados_series.max() or 0.0)
                    chamados_pos = chamados_series[chamados_series > 0]
                    if len(chamados_pos) > 0:
                        max_ref = float(np.percentile(chamados_pos, 95))
                    else:
                        max_ref = 0.0
                    if max_ref <= 0.0:
                        max_ref = max_real

                    def _chamados_style(feature):
                        try:
                            chamados_val = float(feature["properties"].get("__chamados", 0.0))
                        except Exception:
                            chamados_val = 0.0
                        if not (chamados_val > 0 and max_ref > 0):
                            return {
                                "fillColor": "#ffffff",
                                "color": "#1f78b4",
                                "weight": 1.5,
                                "opacity": 0.9,
                                "fillOpacity": 0.0,
                            }
                        t = min(1.0, chamados_val / max_ref)
                        rgba = plt.cm.OrRd(t)
                        fill_color = mcolors.to_hex(rgba, keep_alpha=False)
                        return {
                            "fillColor": fill_color,
                            "color": "#1f78b4",
                            "weight": 1.5,
                            "opacity": 0.9,
                            "fillOpacity": 0.45,
                        }

                    bairros_group = folium.FeatureGroup(
                        name="Chamados por bairro",
                        show=False,
                        overlay=True,
                        control=True,
                    )
                    tooltip_fields = ["__chamados", "__area_km2", "__pct_area_alto"]
                    tooltip_aliases = ["Chamados", "Area (km2)", "% area risco >=3"]
                    if name_col:
                        tooltip_fields = [name_col] + tooltip_fields
                        tooltip_aliases = ["Bairro"] + tooltip_aliases

                    folium.GeoJson(
                        gdf_bairros,
                        name="Chamados por bairro",
                        style_function=_chamados_style,
                        tooltip=folium.GeoJsonTooltip(
                            fields=tooltip_fields,
                            aliases=tooltip_aliases,
                            localize=True,
                        ),
                    ).add_to(bairros_group)
                    bairros_group.add_to(m)
                    chamados_layer_js = bairros_group.get_name()

                    if max_ref > 0:
                        legend_colors = [
                            mcolors.to_hex(plt.cm.OrRd(x), keep_alpha=False)
                            for x in np.linspace(0, 1, 6)
                        ]
                        gradient = ", ".join(legend_colors)
                        max_label = int(round(max_real))
                        cap_label = int(round(max_ref))
                        chamados_legend_html = f"""
                        {{% macro html(this, kwargs) %}}
                        <div id=\"legend_chamados\" style=\"
                            position: fixed;
                            bottom: 20px;
                            left: 190px;
                            width: 170px;
                            z-index: 9999;
                            font-size: 12px;
                            color: #ffffff;
                            background-color: rgba(0,0,0,0.6);
                            border: 2px solid rgba(255,255,255,0.7);
                            border-radius: 5px;
                            padding: 8px;
                            box-shadow: 3px 3px 5px rgba(0,0,0,0.35);
                            display: none;
                        \">
                            <b>Chamados por bairro</b><br>
                            <div style=\"height:12px; margin-top:6px; border:1px solid rgba(255,255,255,0.7); background: linear-gradient(to right, {gradient});\"></div>
                            <div style=\"display:flex; justify-content:space-between; margin-top:4px;\">
                                <span>0</span>
                                <span>{max_label}</span>
                            </div>
                            <div style=\"margin-top:2px; font-size:10px; color:#ffffff; opacity:0.85;\">cap P95: {cap_label}</div>
                        </div>
                        {{% endmacro %}}
                        """
            except Exception as e:
                print(f"Erro ao carregar fronteiras de bairros: {e}")

        if chamados_legend_html and chamados_layer_js:
            chamados_legend_macro = MacroElement()
            chamados_legend_macro._template = Template(chamados_legend_html)
            m.add_child(chamados_legend_macro)

            legend_toggle_template = f"""
            {{% macro script(this, kwargs) %}}
            function _toggleChamadosLegend(show) {{
                const el = document.getElementById('legend_chamados');
                if (!el) return;
                el.style.display = show ? 'block' : 'none';
            }}
            var _chamadosLayer = {chamados_layer_js};
            {map_name}.on('overlayadd', function(e) {{
                if (e.layer === _chamadosLayer) _toggleChamadosLegend(true);
            }});
            {map_name}.on('overlayremove', function(e) {{
                if (e.layer === _chamadosLayer) _toggleChamadosLegend(false);
            }});
            _toggleChamadosLegend({map_name}.hasLayer(_chamadosLayer));
            {{% endmacro %}}
            """
            legend_toggle_macro = MacroElement()
            legend_toggle_macro._template = Template(legend_toggle_template)
            m.add_child(legend_toggle_macro)

        risco_legend_toggle_template = f"""
        {{% macro script(this, kwargs) %}}
        function _toggleRiscoLegend(show) {{
            const el = document.getElementById('legend_risco');
            if (!el) return;
            el.style.display = show ? 'block' : 'none';
        }}
        var _riscoLayer = {risco_layer_js};
        {map_name}.on('overlayadd', function(e) {{
            if (e.layer === _riscoLayer) _toggleRiscoLegend(true);
        }});
        {map_name}.on('overlayremove', function(e) {{
            if (e.layer === _riscoLayer) _toggleRiscoLegend(false);
        }});
        _toggleRiscoLegend({map_name}.hasLayer(_riscoLayer));
        {{% endmacro %}}
        """
        risco_legend_toggle_macro = MacroElement()
        risco_legend_toggle_macro._template = Template(risco_legend_toggle_template)
        m.add_child(risco_legend_toggle_macro)

        # Mostrar a lista expandida para destacar todas as camadas (incluindo HEC-RAS)
        folium.LayerControl(position="topright", collapsed=True, sortLayers=True).add_to(m)

        m.add_child(macro)

        return m._repr_html_()
