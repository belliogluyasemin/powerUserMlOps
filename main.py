from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()



class ml_model_schema(BaseModel):
    medium: str = Field(..., description="Medium", json_schema_extra={"example": "organic"})
    mobile_brand_name: str = Field(..., description="Mobile Brand Name", json_schema_extra={"example": "Samsung"})
    country: str = Field(..., description="Country", json_schema_extra={"example": "United States"})
    category: str = Field(..., description="Category", json_schema_extra={"example": "mobile"})
    sessionCnt: int = Field(..., description="First 15 Day Session Count", json_schema_extra={"example": 10})
    sessionDate: int = Field(..., description="First 15 Day Session Date", json_schema_extra={"example": 10})
    itemBrandCount: int = Field(..., description="First 15 Day Different Item Brands", json_schema_extra={"example": 5})
    itemCatCount: int = Field(..., description="First 15 Day Different Item Category", json_schema_extra={"example": 3})
    viwePromotion: int = Field(..., description="First 15 Day Promotion Viewed", json_schema_extra={"example": 7})
    SelectPromotion: int = Field(..., description="First 15 Day Promotion Selected", json_schema_extra={"example": 4})
    itemViewCnt: int = Field(..., description="First 15 Day Item View Count", json_schema_extra={"example": 20})
    itemSelectCnt: int = Field(..., description="First 15 Day Item Select Count", json_schema_extra={"example": 15})
    paymetInfoAdd: int = Field(..., description="First 15 Day Added Payment Info Count", json_schema_extra={"example": 2})
    shippingInfoAdd: int = Field(..., description="First 15 Day Adding Shipping Info", json_schema_extra={"example": 3})
    ScrollpageLocationCnt: int = Field(..., description="First 15 Day Scrolling Different Page Location", json_schema_extra={"example": 8})
    ScrollpageTitleCnt: int = Field(..., description="First 15 Day Scrolling Different Page Title", json_schema_extra={"example": 9})
    pageViewPageLocationCnt: int = Field(..., description="First 15 Day Viewed Different Page Location", json_schema_extra={"example": 12})
    pageViewPageTitleCnt: int = Field(..., description="First 15 Day Viewed Different Page Title", json_schema_extra={"example": 11})
    itemViews: int = Field(..., description="First 15 Day Item Viewed", json_schema_extra={"example": 25})
    addToCarts: int = Field(..., description="First 15 Day Item Added to Basket", json_schema_extra={"example": 5})
    addToItemId: int = Field(..., description="First 15 Day Added to Basket Different Item", json_schema_extra={"example": 4})
    searchResultViewedCnt: int = Field(..., description="First 15 Day Search Results Viewed Count", json_schema_extra={"example": 6})
    checkOut: int = Field(..., description="First 15 Day Checkout Count", json_schema_extra={"example": 2})
    ecommercePurchases: int = Field(..., description="First 15 Day Purchase Count", json_schema_extra={"example": 3})
    purchaseToViewRate: float = Field(..., description="First 15 Day Item View to Purchase Rate", json_schema_extra={"example": 0.1})
    itemPurchaseName: int = Field(..., description="First 15 Day Added Different Purchased Item", json_schema_extra={"example": 2})
    itemPurchaseQuantity: int = Field(..., description="First 15 Day Added Purchase Quantity", json_schema_extra={"example": 4})
    itemRevenue15: float = Field(..., description="First 15 Day Revenue ($)", json_schema_extra={"example": 100.0})

df = pd.read_pickle("df.pkl")

country_filtered = df[df['country'].isin(["United States", "Canada", "Germany", "India", "France", "Taiwan",'Italy', "Japan","Spain", "China", "Singapore", "South Korea", "Netherlands", 'Turkey'])]
all_filtered = country_filtered.drop(['itemRevenue2','itemRevenue90', 'user_pseudo_id','name','source','cnt','fdate'], axis = 1)
@app.post("/predict/xgboost_adasyn/")
def xgb_adasyn(predict_values: ml_model_schema):
    load_model = pickle.load(open("xgb_adasyn_model.pkl", "rb"))

    input_df = pd.DataFrame(
        [predict_values.model_dump().values()],
        columns=predict_values.model_dump().keys()
    )
    new_df = pd.concat([all_filtered,input_df], axis = 0)
    gdp = pd.read_pickle("gdp.pkl")
    gdp = gdp[['GDP per capita, current prices\n (U.S. dollars per capita)','2020','2021']]
    gdp = gdp.rename(columns= {'GDP per capita, current prices\n (U.S. dollars per capita)':'country'})
    gdp = gdp.rename(columns= {'2020':'gdp_2020_value'})
    gdp = gdp.rename(columns= {'2021':'gdp_2021_value'})
    merged_df = pd.merge(new_df, gdp, on='country',  how='left')
    merged_df['gdp_2020_value'] = merged_df['gdp_2020_value'].astype(float)
    merged_df['gdp_2021_value'] = merged_df['gdp_2021_value'].astype(float)
    merged_df['Avg_gdp'] =  merged_df[['gdp_2020_value','gdp_2021_value']].mean(axis = 1)
    merged_df.dropna(inplace = True)
    merged_df.loc[:, 'LogGDP'] = np.log(merged_df['Avg_gdp'])
    dummies_input = pd.get_dummies(merged_df[['medium', 'mobile_brand_name', 'country', 'category']], drop_first=True, dtype=int)
    dummies_input = pd.concat([merged_df,dummies_input], axis = 1)
    x_input = dummies_input.drop(['medium','mobile_brand_name','country','category'], axis = 1,)
    xc_input =  x_input.copy()
    xc_input['perBasket'] = xc_input.apply(
        lambda row: row['itemPurchaseQuantity'] / row['ecommercePurchases'] if row['ecommercePurchases'] != 0 else 0,
        axis=1
    )
    ##xc_input = pd.DataFrame(xc_input)
    xc_input = xc_input.rename(columns={"medium_(none)": "medium_none", "medium_<Other>": "medium_Other"})
    test_df = x_input.iloc[-1].to_frame().T
    test_df_c = xc_input.iloc[-1].to_frame().T

    print(test_df_c)
    predict = load_model.predict(test_df_c)
    return {"Predict": int(predict[0])}

## triger fastApi 
##run local port 
##python -m uvicorn main:app --reload