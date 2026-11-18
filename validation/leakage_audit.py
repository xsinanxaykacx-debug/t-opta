def audit_feature_columns(feature_columns, forbidden_columns):
    return sorted(set(feature_columns) & set(forbidden_columns))
