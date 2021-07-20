import os
import pandas as pd
import json
import scipy.sparse as sp_sparse
import tables
import umap.umap_ as umap

base_path = "../../data/"


def read_h5_file(version):
    file_path = os.path.join(base_path, "{}/filtered_feature_bc_matrix.h5".format(version))
    f = tables.open_file(file_path, 'r')

    # matrix
    mat_group = f.get_node(f.root, 'matrix')
    data = getattr(mat_group, 'data').read()
    indices = getattr(mat_group, 'indices').read()
    indptr = getattr(mat_group, 'indptr').read()
    shape = getattr(mat_group, 'shape').read()
    matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

    # features
    feature_group = f.get_node(mat_group, 'features')
    feature_ids = getattr(feature_group, 'id').read()
    feature_names = getattr(feature_group, 'name').read()
    feature_types = getattr(feature_group, 'feature_type').read()

    # barcodes
    barcodes = f.get_node(mat_group, 'barcodes').read()

    # processing
    mat_lists = matrix.tocsr().toarray().tolist()
    df_features = pd.DataFrame({'id': [x.decode('UTF-8') for x in feature_ids],
                                'name': [x.decode('UTF-8') for x in feature_names],
                                'type': [x.decode('UTF-8') for x in feature_types]})
    list_barcodes = [d.decode('UTF-8') for d in barcodes]

    f.close()
    return mat_lists, df_features, list_barcodes


def get_base_result(version, df_features, mat_lists, list_barcodes):
    # read spatial
    spatial_path = os.path.join(base_path, "{}/spatial".format(version))
    df_spatial = pd.read_csv(os.path.join(spatial_path, "tissue_positions_list.csv"),
                         names=["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres",
                                "pxl_row_in_fullres"])

    # process spatial
    df_position = df_spatial[df_spatial['in_tissue'] == 1]
    df_position = df_position.drop(columns=['in_tissue'])
    df_position['array_col'] = df_position['array_col'].apply(lambda x: int(x / 2) if (x % 2 == 0) else int((x + 1) / 2))
    df_position = df_position.rename(columns={'array_row': 'row', 'array_col': 'col'})

    # concatenate df_genes
    df_gene = pd.DataFrame({'version': version, 'barcode': list_barcodes})
    feature_names = df_features['name']
    for idx in range(19500, 20001): #sample
        feature_name = feature_names[idx]
        list_expressions = mat_lists[idx]
        df_gene[feature_name] = list_expressions
    for feature_name in ['SNAP25', 'MOBP', 'PCP4', 'FABP7', 'PVALB', 'CCK', 'ENC1',
                         'CUX2', 'ADCYAP1', 'RORB', 'NTNG2', 'MBP']: # important genes
        if feature_name in df_features['name'].values.tolist():
            idx = df_features.index[df_features['name'] == feature_name].tolist()[0]
            list_expressions = mat_lists[idx]
            df_gene[feature_name] = list_expressions

    # processing
    df_result = df_gene.merge(df_position,  how='inner', on='barcode')

    # output
    output_path = os.path.join(base_path, "{}/processed/matrix.csv".format(version))
    df_result.to_csv(output_path, index=False, encoding="utf-8")
    return df_result


def transfer_img_pixels(version, df_result):
    file_path = os.path.join(base_path, "{}/spatial/scalefactors_json.json".format(version))
    with open(file_path) as f:
        scale_factors = json.loads(f.read())
        lowres_factor = scale_factors['tissue_lowres_scalef']
        highres_factor = scale_factors['tissue_hires_scalef']

        # rows
        list_row_pxl = df_result['pxl_row_in_fullres'].values.tolist()
        list_row_pxl_lowres = [round(k * lowres_factor) for k in list_row_pxl]
        list_row_pxl_highres = [round(k * highres_factor) for k in list_row_pxl]

        # columns
        list_col_pxl = df_result['pxl_col_in_fullres'].values.tolist()
        list_col_pxl_lowres = [round(k * lowres_factor) for k in list_col_pxl]
        list_col_pxl_highres = [round(k * highres_factor) for k in list_col_pxl]

        # lowres
        df_result['pxl_col_in_lowres'] = list_col_pxl_lowres
        df_result['pxl_row_in_lowres'] = list_row_pxl_lowres

        # highres
        df_result['pxl_col_in_highres'] = list_col_pxl_highres
        df_result['pxl_row_in_highres'] = list_row_pxl_highres

    output_path = os.path.join(base_path, "{}/processed/spatial.csv".format(version))
    df_result.to_csv(output_path, index=False, encoding="utf-8")
    return df_result


def add_cluster_columns(version, df_spatial):
    cluster_path_1 = os.path.join(base_path, "{}/cluster_labels.csv".format(version))
    cluster_path_2 = os.path.join(base_path, "{VER}/analysis/clustering/".format(VER=version)) + \
                     "kmeans_{}_clusters/clusters.csv"
    list_columns = ["SpatialDE_PCA", "SpatialDE_pool_PCA", "HVG_PCA", "pseudobulk_PCA", "markers_PCA",
                          "SpatialDE_UMAP",
                          "SpatialDE_pool_UMAP", "HVG_UMAP", "pseudobulk_UMAP", "markers_UMAP", "SpatialDE_PCA_spatial",
                          "SpatialDE_pool_PCA_spatial", "HVG_PCA_spatial", "pseudobulk_PCA_spatial",
                          "markers_PCA_spatial",
                          "SpatialDE_UMAP_spatial", "SpatialDE_pool_UMAP_spatial", "HVG_UMAP_spatial",
                          "pseudobulk_UMAP_spatial",
                          "markers_UMAP_spatial"]

    if os.path.isfile(cluster_path_1):
        df_cluster = pd.read_csv(cluster_path_1)
        df_cluster['barcode'] = df_cluster['key'].apply(lambda x: x.split('_')[1])
        df_cluster = df_cluster.fillna('NA')

        list_idx = []
        for x in df_cluster['ground_truth']:
            if x == 'WM':
                list_idx.append(7)
            elif x == 'NA':
                list_idx.append(8)
            else:
                x = x.split('_')[1]
                list_idx.append(x)

        for i in range(2, 11):
            if (i == 7):
                df_cluster['cluster_{}'.format(str(i))] = list_idx
            else:
                df_cluster['cluster_{}'.format(str(i))] = 1

        df_cluster = df_cluster.drop(columns=['key', 'ground_truth'])
        df_spatial = df_spatial.merge(df_cluster, how='inner', on='barcode')

    elif os.path.isfile(cluster_path_2.format(2)):
        for i in range(2, 11):
           df_cluster = pd.read_csv(cluster_path_2.format(i))
           df_cluster = df_cluster.rename(columns={'Barcode': 'barcode', 'Cluster': 'cluster_{}'.format(str(i))})
           df_spatial = df_spatial.merge(df_cluster, how='inner', on='barcode')

        for k in list_columns:
            df_spatial[k] = 1

    output_path = os.path.join(base_path, "{}/processed/spatial_cluster.csv".format(version))
    df_spatial.to_csv(output_path, index=False)
    return df_spatial


def add_umap_columns(version, df):
    df_umap = df.drop(columns=['version', 'barcode', 'row', 'col', 'pxl_col_in_fullres',
                               'pxl_row_in_fullres', 'pxl_col_in_lowres', 'pxl_row_in_lowres',
                               'pxl_col_in_highres', 'pxl_row_in_highres'])

    # get embedding
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df_umap)
    df_embedding = pd.DataFrame(embedding)
    df_embedding = df_embedding.rename(columns={0: 'umap_x', 1: 'umap_y'})
    df_result = pd.concat([df_cluster, df_embedding], axis=1)

    # output
    output_path = os.path.join(base_path, "{}/processed/spatial_embedding.csv".format(version))
    df_result.to_csv(output_path.format(version), index=False)
    return df_result


if __name__ == '__main__':
    list_versions = ['human_DLPFC/151510', 'human_DLPFC/151671',
                     'human_DLPFC/151674','human_DLPFC/151508', 'human_DLPFC/151669', 'human_DLPFC/151672',
                     'human_DLPFC/151675', 'human_DLPFC/151509', 'human_DLPFC/151670', 'human_DLPFC/151676']
    #'151507', '151510', '151671','151674', '151508', '151669', '151672', '151675', '151509', '151670', '151673', '151676'

    for version in list_versions:
        # make directory
        output_path = os.path.join(base_path, "{}/processed".format(version))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        mat_lists, df_features, list_barcodes = read_h5_file(version)
        print("VERSION: %s." % version)

        df_result = get_base_result(version, df_features, mat_lists, list_barcodes)
        print("base formation completed.")

        df_spatial = transfer_img_pixels(version, df_result)
        print("pixel transformation completed.")

        df_cluster = add_cluster_columns(version, df_spatial)
        print("cluster combination completed.")

        df_embedding = add_umap_columns(version, df_result)
        print("FINISHED: spatial_embedding has %d rows and %d columns" % (df_embedding.shape[0], df_embedding.shape[1]))