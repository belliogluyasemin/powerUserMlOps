from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_xgb_adasyn():
    payload = {
        "sessionCnt": 2,
        "sessionDate": 1,
        "itemBrandCount": 2,
        "itemCatCount": 3,
        "viwePromotion": 3,
        "SelectPromotion": 0,
        "itemViewCnt": 23,
        "itemSelectCnt": 0,
        "paymetInfoAdd": 0,
        "shippingInfoAdd": 0,
        "ScrollpageLocationCnt": 1,
        "ScrollpageTitleCnt": 1,
        "pageViewPageLocationCnt": 5,
        "pageViewPageTitleCnt": 5,
        "itemViews": 9,
        "addToCarts": 0,
        "addToItemId": 0,
        "searchResultViewedCnt": 0,
        "checkOut": 0,
        "ecommercePurchases": 0,
        "purchaseToViewRate": 0.0,
        "itemPurchaseName": 0,
        "itemPurchaseQuantity": 0,
        "itemRevenue15": 0.0,
        "gdp_2020_value": 64367.435,
        "gdp_2021_value": 70995.794,
        "Avg_gdp": 67681.6145,
        "LogGDP": 11.122570,
        "medium": "organic",
        "mobile_brand_name": "Apple",
        "country": "United States",
        "category": "mobile",
        "medium_none": 0,
        "medium_Other": 0,
        "medium_cpc": 0,
        "medium_organic": 1,
        "medium_referral": 0,
        "mobile_brand_name_Apple": 1,
        "mobile_brand_name_Google": 0,
        "mobile_brand_name_Huawei": 0,
        "mobile_brand_name_Microsoft": 0,
        "mobile_brand_name_Mozilla": 0,
        "mobile_brand_name_Samsung": 0,
        "mobile_brand_name_Xiaomi": 0,
        "country_China": 0,
        "country_France": 0,
        "country_Germany": 0,
        "country_India": 0,
        "country_Italy": 0,
        "country_Japan": 0,
        "country_Netherlands": 0,
        "country_Singapore": 0,
        "country_South Korea": 0,
        "country_Spain": 0,
        "country_Taiwan": 0,
        "country_Turkey": 0,
        "country_United States": 1,
        "category_mobile": 0,
        "category_tablet": 0,
        "perBasket": 0
    }
    
    response = client.post("/predict/xgboost_adasyn/", json=payload)
    
    print(response.json())  # Hata mesajını görmek için

    assert response.status_code == 200
    assert "Predict" in response.json()
    assert response.json()["Predict"] in [0, 1]

