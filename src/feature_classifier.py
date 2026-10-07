def classify_features(df):
    """
    Classify dataset columns into meaningful categories.
    """

    categories = {
        "identifier": [],
        "temporal": [],
        "spatial": [],
        "crime": [],
        "numerical": [],
        "categorical": []
    }

    for column in df.columns:

        col = column.lower().strip()

       
        if col in [
            "id",
            "case_number",
            "case_id"
        ]:
            categories["identifier"].append(column)

       
        elif col in [
            "date",
            "datetime",
            "timestamp",
            "year",
            "month",
            "day",
            "hour"
        ]:
            categories["temporal"].append(column)

    
        elif col in [
            "latitude",
            "longitude",
            "lat",
            "lon",
            "lng",
            "beat",
            "district",
            "ward",
            "community_area"
        ]:
            categories["spatial"].append(column)

  
        elif col in [
            "primary_type",
            "description",
            "location_description",
            "arrest",
            "domestic",
            "iucr",
            "fbi_code"
        ]:
            categories["crime"].append(column)

        elif df[column].dtype.kind in "iuf":

            categories["numerical"].append(column)

        else:

            categories["categorical"].append(column)

    return categories  