# pylint: disable=invalid-name, line-too-long
"""
Feature Engineering for Housing Prices Dataset
"""
import pandas as pd

def impute_most_frequent(df, cols, value):
    """
    Impute missing values in the specified column with the most frequent value.
    """
    cols = [cols] if isinstance(cols, str) else cols

    replacements = {}
    for col in cols:
        most_frequent = df.loc[df[col] != value, col].mode()[0]
        replacements[col] = most_frequent

    replacement_dict = {col: {value: freq} for col, freq in replacements.items()}
    return df.replace(replacement_dict, inplace=True)



def feature_eng(df, cols, defaults, code_mappings, drop=True):
    """
    Feature engineering function to create new features based on categorical codes.
    """
    # print(f"Feature engineering {cols}...")

    cols = [cols] if isinstance(cols, str) else cols
    prefix = f"{cols[0]}_"

    new_features = {}
    for feature, default_value in defaults.items():
        feature_name = f"{prefix}{feature}"
        new_features[feature_name] = pd.Series(default_value, index=df.index)

    column_masks = {}
    for column in cols:
        impute_most_frequent(df, column, pd.NA)
        column_masks[column] = {code: df[column].eq(code) for code in code_mappings}

    for code, features in code_mappings.items():
        combined_mask = column_masks[cols[0]][code]
        for column in cols[1:]:
            combined_mask = combined_mask | column_masks[column][code]

        for feature, value in features.items():
            feature_name = f"{prefix}{feature}"
            new_features[feature_name] = new_features[feature_name].mask(combined_mask, value)

    for column_name, column_values in new_features.items():
        df[column_name] = column_values

    if drop:
        df.drop(columns=cols, inplace=True)

    return df



def fill_na(df, cols, value):
    """
    Fill NaN values in the specified columns with 0.
    """
    cols = [cols] if isinstance(cols, str) else cols
    df[cols] = df[cols].fillna(value)


def feature_engineering(df):
    """
    Perform feature engineering on the given DataFrame.
    """
    df = df.copy()
    print(
        f"""Feature engineering housing transactions: """
        f"""rows {len(df)}, """
        f"""columns before {len(df.columns)}, """,
        end=""
        )

    #MSSubClass
    MSSubClass_defaults = {
            'stories': -10,
            'age': 'unspecified',
            'attic_finished': False,
            'is_pud': False,
            'is_duplex': False,
            'is_family_conversion': False,
            'is_split_level': False,
            'type': 'unspecified'
        }
    MSSubClass_code_mappings = {
        20: {'stories': 10, 'age': '1946 & newer', 'type': 'Standard'},  # 1 story
        30: {'stories': 10, 'age': '1945 & older', 'type': 'Standard'},  # 1 story
        40: {'stories': 10, 'age': 'All ages', 'attic_finished': True, 'type': 'Standard'},  # 1 story w/ finished attic
        45: {'stories': 15, 'age': 'All ages', 'attic_finished': False, 'type': 'Standard'},  # 1.5 story, unfinished
        50: {'stories': 15, 'age': 'All ages', 'attic_finished': True, 'type': 'Standard'},  # 1.5 story, finished
        60: {'stories': 20, 'age': '1946 & newer', 'type': 'Standard'},  # 2 story
        70: {'stories': 20, 'age': '1945 & older', 'type': 'Standard'},  # 2 story
        75: {'stories': 25, 'age': 'All ages', 'type': 'Standard'},  # 2.5 story
        80: {'stories': 20, 'age': 'All ages', 'is_split_level': True, 'type': 'Standard'},  # Split/multi-level (treated as 2 story)
        85: {'stories': 20, 'age': 'All ages', 'is_split_level': True, 'type': 'Standard'},  # Split foyer (treated as 2 story)
        90: {'stories': 20, 'age': 'All ages', 'is_duplex': True, 'type': 'Duplex'},  # Duplex (treated as 2 story)
        120: {'stories': 10, 'age': '1946 & newer', 'is_pud': True, 'type': 'PUD'},  # 1 story PUD
        150: {'stories': 15, 'age': 'All ages', 'is_pud': True, 'type': 'PUD'},  # 1.5 story PUD
        160: {'stories': 20, 'age': '1946 & newer', 'is_pud': True, 'type': 'PUD'},  # 2 story PUD
        180: {'stories': 20, 'age': 'All ages', 'is_pud': True, 'is_split_level': True, 'type': 'PUD'},  # Multi-level PUD
        190: {'stories': 20, 'age': 'All ages', 'is_family_conversion': True, 'type': '2 Family Conversion'}  # 2 family conversion
        }
    feature_eng(df, cols='MSSubClass', defaults=MSSubClass_defaults, code_mappings=MSSubClass_code_mappings)
    impute_most_frequent(df, 'MSSubClass_stories', -10)

    #MSZoning
    MSZoning_defaults = {
        'is_residential': False,
        'is_commercial': False,
        'is_industrial': False,
        'is_agricultural': False,
        'density': -1,
        'is_park': False,
        'is_floating_village': False
    }
    MSZoning_code_mappings = {
        'A': {'is_agricultural': True},
        'C': {'is_commercial': True},
        'FV': {'is_residential': True, 'is_floating_village': True},
        'I': {'is_industrial': True},
        'RH': {'is_residential': True, 'density': 3},
        'RL': {'is_residential': True, 'density': 1},
        'RP': {'is_residential': True, 'density': 1, 'is_park': True},
        'RM': {'is_residential': True, 'density': 2}
    }
    feature_eng(df, cols='MSZoning', defaults=MSZoning_defaults, code_mappings=MSZoning_code_mappings)

    #Street, Alley
    Street_defaults = {'quality_score': 0}
    Street_code_mappings = {
        'Pave': {'quality_score': 2},
        'Grvl': {'quality_score': 1},
        'NA': {'quality_score': 0}
        }
    feature_eng(df, cols='Street', defaults=Street_defaults, code_mappings=Street_code_mappings)
    fill_na(df, 'Alley', 'NA')
    feature_eng(df, cols='Alley', defaults=Street_defaults, code_mappings=Street_code_mappings)

    #LotShape
    LotShape_defaults = {
        'is_regular': False,
        'irregularity_level': 0
    }
    LotShape_code_mappings = {
        'Reg': {'is_regular': True, 'irregularity_level': 0},
        'IR1': {'is_regular': False, 'irregularity_level': 1},
        'IR2': {'is_regular': False, 'irregularity_level': 2},
        'IR3': {'is_regular': False, 'irregularity_level': 3}
    }
    feature_eng(df, cols='LotShape', defaults=LotShape_defaults, code_mappings=LotShape_code_mappings)

    #LandContour
    LandContour_defaults = {
        'is_flat': False,
        'is_banked': False,
        'is_hillside': False,
        'is_depression': False,
        'elevation_change': 0,
        'quality_score': 0
    }

    LandContour_code_mappings = {
        'Lvl': {'is_flat': True, 'elevation_change': 0, 'quality_score': 3},
        'Bnk': {'is_banked': True, 'elevation_change': 2, 'quality_score': 1},
        'HLS': {'is_hillside': True, 'elevation_change': 1 , 'quality_score': 2},
        'Low': {'is_depression': True, 'elevation_change': -1, 'quality_score': 1},
    }
    feature_eng(df, cols='LandContour', defaults=LandContour_defaults, code_mappings=LandContour_code_mappings)

    #Utilities
    Utilities_defaults = {
        'has_electricity': False,
        'has_gas': False,
        'has_water': False,
        'has_sewer': False,
        'utility_count': 0
    }

    Utilities_code_mappings = {
        'AllPub': {'has_electricity': True, 'has_gas': True, 'has_water': True, 'has_sewer': True, 'utility_count': 4},
        'NoSewr': {'has_electricity': True, 'has_gas': True, 'has_water': True, 'utility_count': 3},
        'NoSeWa': {'has_electricity': True, 'has_gas': True, 'utility_count': 2},
        'ELO': {'has_electricity': True, 'utility_count': 1}
    }
    feature_eng(df, cols='Utilities', defaults=Utilities_defaults, code_mappings=Utilities_code_mappings)

    #LotConfig
    LotConfig_defaults = {
        'is_inside': False,
        'is_corner': False,
        'is_cul_de_sac': False,
        'frontage_count': 1
    }

    LotConfig_code_mappings = {
        'Inside': {'is_inside': True, 'frontage_count': 1},
        'Corner': {'is_corner': True, 'frontage_count': 2},
        'CulDSac': {'is_cul_de_sac': True, 'frontage_count': 1},
        'FR2': {'frontage_count': 2},
        'FR3': {'frontage_count': 3}
    }
    feature_eng(df, cols='LotConfig', defaults=LotConfig_defaults, code_mappings=LotConfig_code_mappings)

    #LandSlope
    LandSlope_defaults = {
        'is_gentle': False,
        'is_moderate': False,
        'is_severe': False,
        'slope_intensity': 0
    }

    LandSlope_code_mappings = {
        'Gtl': {'is_gentle': True, 'slope_intensity': 1},
        'Mod': {'is_moderate': True, 'slope_intensity': 2},
        'Sev': {'is_severe': True, 'slope_intensity': 3}
    }
    feature_eng(df, cols='LandSlope', defaults=LandSlope_defaults, code_mappings=LandSlope_code_mappings)

    #Condition1, Condition2
    Condition_defaults = {
        'near_arterial': False,
        'near_feeder': False,
        'near_railroad': False,
        'near_positive_feature': False,
        'is_normal': False,
        'railorad_proximity': 5,
        'off_site_proximity': 5,
        'has_ns_railroad': False,
        'has_ew_railroad': False
    }

    # Common code mappings for both Condition1 and Condition2
    Condition_code_mappings = {
        'Artery': {'near_arterial': True},
        'Feedr': {'near_feeder': True},
        'Norm': {'is_normal': True},
        'RRNn': {'near_railroad': True, 'has_ns_railroad': True, 'railorad_proximity': 1},
        'RRAn': {'near_railroad': True, 'has_ns_railroad': True, 'railorad_proximity': 2},
        'PosN': {'near_positive_feature': True, 'off_site_proximity': 1},
        'PosA': {'near_positive_feature': True, 'off_site_proximity': 2},
        'RRNe': {'near_railroad': True, 'has_ew_railroad': True, 'railorad_proximity': 1},
        'RRAe': {'near_railroad': True, 'has_ew_railroad': True, 'railorad_proximity': 2}
    }
    feature_eng(df, cols=['Condition1', 'Condition2'], defaults=Condition_defaults, code_mappings=Condition_code_mappings)

    #HomeStyle
    HouseStyle_defaults = {
        'stories': 0,
        'finished_upper': False,
        'is_split_foyer': False,
        'is_split_level': False,
        'type': 'unspecified'
    }
    HouseStyle_code_mappings = {
        '1Story': {'stories': 10, 'finished_upper': False, 'type': 'Standard'},
        '1.5Fin': {'stories': 15, 'finished_upper': True, 'type': 'Standard'},
        '1.5Unf': {'stories': 15, 'finished_upper': False, 'type': 'Standard'},
        '2Story': {'stories': 20, 'finished_upper': False, 'type': 'Standard'},
        '2.5Fin': {'stories': 25, 'finished_upper': True, 'type': 'Standard'},
        '2.5Unf': {'stories': 25, 'finished_upper': False, 'type': 'Standard'},
        'SFoyer': {'stories': 15, 'is_split_foyer': True, 'type': 'Split'},
        'SLvl': {'stories': 15, 'is_split_level': True, 'type': 'Split'}
    }
    feature_eng(df, cols='HouseStyle', defaults=HouseStyle_defaults, code_mappings=HouseStyle_code_mappings)
    impute_most_frequent(df, 'HouseStyle_stories', 0)

    #Exterior1st
    Exterior_defaults = {
        'material_type': 'unspecified',
        'is_wood_based': False,
        'is_masonry': False,
        'is_siding': False,
        'is_shingle': False,
        'is_stucco': False,
        'is_premium': False,
        'durability': -1
    }
    Exterior_code_mappings = {
        'AsbShng': {'material_type': 'asbestos', 'is_shingle': True, 'durability': 1},
        'AsphShn': {'material_type': 'asphalt', 'is_shingle': True, 'durability': 1},
        'BrkComm': {'material_type': 'brick','is_masonry': True, 'durability': 3, 'is_premium': False},
        'BrkFace': {'material_type': 'brick', 'is_masonry': True, 'durability': 3, 'is_premium': True},
        'CBlock': {'material_type': 'cinder_block', 'is_masonry': True, 'durability': 3, 'is_premium': False},
        'CemntBd': {'material_type': 'cement', 'is_siding': True, 'durability': 3},
        'HdBoard': {'material_type': 'hardboard', 'is_wood_based': True, 'is_siding': True, 'durability': 2},
        'ImStucc': {'material_type': 'imitation_stucco','is_stucco': True, 'durability': 2},
        'MetalSd': {'material_type': 'metal', 'is_siding': True, 'durability': 3},
        'Other': {'material_type': 'other', 'durability': 2},
        'Plywood': {'material_type': 'plywood', 'is_wood_based': True, 'durability': 2},
        'PreCast': {'material_type': 'precast', 'is_masonry': True,'durability': 3,'is_premium': True},
        'Stone': {'material_type': 'stone', 'is_masonry': True,'durability': 3,'is_premium': True},
        'Stucco': {'material_type': 'stucco', 'is_stucco': True,'durability': 3},
        'VinylSd': {'material_type': 'vinyl', 'is_siding': True,'durability': 2},
        'Wd Sdng': {'material_type': 'wood', 'is_wood_based': True, 'is_siding': True,'durability': 2},
        'WdShing': {'material_type': 'wood', 'is_wood_based': True, 'is_shingle': True,'durability': 2}
    }
    feature_eng(df, cols=['Exterior1st', 'Exterior2nd'], defaults=Exterior_defaults, code_mappings=Exterior_code_mappings)
    impute_most_frequent(df, 'Exterior1st_material_type', 'unspecified')
    #impute_most_frequent(df, 'Exterior1st_durability', -1)

    #MasVnrType
    MasVnrType_defaults = {
        'has_veneer': False,
        'material_type': 'none',
        'is_brick': False,
        'is_stone': False,
        'is_cinder_block': False,
        'is_premium': False,
        'durability': -1
    }
    MasVnrType_code_mappings = {
        'BrkCmn': {'has_veneer': True, 'material_type': 'brick', 'is_brick': True, 'durability': 3, 'is_premium': False},
        'BrkFace': {'has_veneer': True, 'material_type': 'brick', 'is_brick': True, 'durability': 3, 'is_premium': True},
        'CBlock': {'has_veneer': True, 'material_type': 'cinder_block', 'is_cinder_block': True, 'durability': 3, 'is_premium': False},
        'None': {'has_veneer': False, 'material_type': 'none', 'durability': 1},
        'Stone': {'has_veneer': True, 'material_type': 'stone', 'is_stone': True, 'durability': 3, 'is_premium': True}
    }
    feature_eng(df, cols='MasVnrType', defaults=MasVnrType_defaults, code_mappings=MasVnrType_code_mappings)

    #ExterQual, ExterCond
    ExterQual_defaults = { 'quality_score': 0}
    ExterQual_code_mappings = {
        'Ex': {'quality_score': 5},
        'Gd': {'quality_score': 4},
        'TA': {'quality_score': 3},
        'Fa': {'quality_score': 2},
        'Po': {'quality_score': 1}
    }
    feature_eng(df, cols='ExterQual', defaults=ExterQual_defaults, code_mappings=ExterQual_code_mappings)
    feature_eng(df, cols='ExterCond', defaults=ExterQual_defaults, code_mappings=ExterQual_code_mappings)

    #Foundation
    Foundation_defaults = {'quality_score': 0}
    Foundation_code_mappings = {
        'PConc': {'quality_score': 5},
        'CBlock': {'quality_score': 4},
        'BrkTil': {'quality_score': 3},
        'Stone': {'quality_score': 3},
        'Slab': {'quality_score': 2},
        'Wood': {'quality_score': 1},
        'unspecified': {'quality_score': 0} #3
    }
    feature_eng(df, cols='Foundation', defaults=Foundation_defaults, code_mappings=Foundation_code_mappings)

    #BsmtQual, BsmtCond
    BsmtQual_defaults = {'quality_score': 0}
    BsmtQual_code_mappings = {
        'Ex': {'quality_score': 5},
        'Gd': {'quality_score': 4},
        'TA': {'quality_score': 3},
        'Fa': {'quality_score': 2},
        'Po': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    feature_eng(df, cols='BsmtQual', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)
    feature_eng(df, cols='BsmtCond', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #BsmtExposure
    BsmtExposure_defaults = {'quality_score': 0}
    BsmtExposure_code_mappings = {
        'Gd': {'quality_score': 4},
        'Av': {'quality_score': 3},
        'Mn': {'quality_score': 2},
        'No': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    feature_eng(df, cols='BsmtExposure', defaults=BsmtExposure_defaults, code_mappings=BsmtExposure_code_mappings)

    #BsmntFinType
    BsmtFinType1_defaults = {'quality_score': 0}
    BsmtFinType1_code_mappings = {
        'GLQ': {'quality_score': 6},
        'ALQ': {'quality_score': 5},
        'BLQ': {'quality_score': 4},
        'Rec': {'quality_score': 3},
        'LwQ': {'quality_score': 2},
        'Unf': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    feature_eng(df, cols='BsmtFinType1', defaults=BsmtFinType1_defaults, code_mappings=BsmtFinType1_code_mappings)
    feature_eng(df, cols='BsmtFinType2', defaults=BsmtFinType1_defaults, code_mappings=BsmtFinType1_code_mappings)

    #Heating
    Heating_defaults = {'quality_score': 0}
    Heating_code_mappings = {
        'GasA': {'quality_score': 6},
        'GasW': {'quality_score': 5},
        'OthW': {'quality_score': 4},
        'Grav': {'quality_score': 3},
        'Wall': {'quality_score': 2},
        'Floor': {'quality_score': 1}
    }
    feature_eng(df, cols='Heating', defaults=Heating_defaults, code_mappings=Heating_code_mappings, drop=False)

    #HeatingQC
    feature_eng(df, cols='HeatingQC', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #CentralAir
    CentralAir_defaults = {'AC': False}
    CentralAir_code_mappings = {
        'Y': {'AC': True},
        'N': {'AC': False}
    }
    feature_eng(df, cols='CentralAir', defaults=CentralAir_defaults, code_mappings=CentralAir_code_mappings)

    #Electrical
    Electrical_defaults = {'quality_score': 0}
    Electrical_code_mappings = {
        'SBrkr': {'quality_score': 5},
        'FuseA': {'quality_score': 4},
        'Mix': {'quality_score': 3},
        'FuseF': {'quality_score': 2},
        'FuseP': {'quality_score': 1}
    }
    feature_eng(df, cols='Electrical', defaults=Electrical_defaults, code_mappings=Electrical_code_mappings, drop=False) # drop

    #KitchenQual
    feature_eng(df, cols='KitchenQual', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #Functional
    Functional_defaults = {'quality_score': 0}
    Functional_code_mappings = {
        'Typ': {'quality_score': 8},
        'Min1': {'quality_score': 7},
        'Min2': {'quality_score': 6},
        'Mod': {'quality_score': 5},
        'Maj1': {'quality_score': 4},
        'Maj2': {'quality_score': 3},
        'Sev': {'quality_score': 2},
        'Sal': {'quality_score': 1}
    }
    feature_eng(df, cols='Functional', defaults=Functional_defaults, code_mappings=Functional_code_mappings, drop=False) # drop

    #FireplaceQu
    feature_eng(df, cols='FireplaceQu', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #GarageType
    GarageType_defaults = {'quality_score': 0}
    GarageType_code_mappings = {
        '2Types': {'quality_score': 4},
        'BuiltIn': {'quality_score': 6},
        'Attchd': {'quality_score': 5},
        'Basment': {'quality_score': 3},
        'Detchd': {'quality_score': 2},
        'CarPort': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    fill_na(df, 'GarageType', 'NA')
    feature_eng(df, cols='GarageType', defaults=GarageType_defaults, code_mappings=GarageType_code_mappings, drop=False)

    #GarageFinish
    GarageFinish_defaults = {'quality_score': 0}
    GarageFinish_code_mappings = {
        'Fin': {'quality_score': 3},
        'RFn': {'quality_score': 2},
        'Unf': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    feature_eng(df, cols='GarageFinish', defaults=GarageFinish_defaults, code_mappings=GarageFinish_code_mappings)

    #GarageQual, GarageCond
    feature_eng(df, cols='GarageQual', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)
    feature_eng(df, cols='GarageCond', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #PavedDrive
    PavedDrive_defaults = {'quality_score': 0}
    PavedDrive_code_mappings = {
        'Y': {'quality_score': 3},
        'P': {'quality_score': 2},
        'N': {'quality_score': 1}
    }
    feature_eng(df, cols='PavedDrive', defaults=PavedDrive_defaults, code_mappings=PavedDrive_code_mappings)

    #PoolQC
    feature_eng(df, cols='PoolQC', defaults=BsmtQual_defaults, code_mappings=BsmtQual_code_mappings)

    #Fence
    Fence_defaults = {'quality_score': 0}
    Fence_code_mappings = {
        'GdPrv': {'quality_score': 4},
        'GdWo': {'quality_score': 3},
        'MnPrv': {'quality_score': 2},
        'MnWw': {'quality_score': 1},
        'NA': {'quality_score': 0}
    }
    feature_eng(df, cols='Fence', defaults=Fence_defaults, code_mappings=Fence_code_mappings, drop=False) # drop

    #SaleType
    fill_na(df,'SaleType','WD')
    SaleType_defaults = {'quality_score': 0}
    SaleType_code_mappings = {
        'WD': {'quality_score': 6},
        'New': {'quality_score': 5},
        'CWD': {'quality_score': 4},
        'VWD': {'quality_score': 3},
        'Con': {'quality_score': 2},
        'ConLw': {'quality_score': 2},
        'ConLI': {'quality_score': 2},
        'ConLD': {'quality_score': 2},
        'COD': {'quality_score': 1},
        'Oth': {'quality_score': 1}
    }
    feature_eng(df, cols='SaleType', defaults=SaleType_defaults, code_mappings=SaleType_code_mappings, drop=False)

    #SaleCondition
    SaleCondition_defaults = {'quality_score': 0}
    SaleCondition_code_mappings = {
        'Normal': {'quality_score': 6},
        'Partial': {'quality_score': 5},
        'AdjLand': {'quality_score': 4},
        'Alloca': {'quality_score': 3},
        'Family': {'quality_score': 2},
        'Abnorml': {'quality_score': 1}
    }
    feature_eng(df, cols='SaleCondition', defaults=SaleCondition_defaults, code_mappings=SaleCondition_code_mappings, drop=False) #drop

    df=df.copy()

    print("columns after:", len(df.columns))

    fill_na(df,['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea'],0)
    fill_na(df,['LotFrontage','MasVnrArea'],0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    fill_na(df,'MiscFeature','NA')

    # New features
    df['Age']=df['YrSold']-df['YearBuilt']
    df['YrsRemodAdd']=df['YrSold']-df['YearRemodAdd']

    df['TotalBuildingArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['LowQualFinSF'] + df['TotalBsmtSF'] + df['GarageArea']
    df['ExteriorFeaturesArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['PoolArea']

    return df
