#!/usr/bin/env python

import time
import logging
from train_ticket_auto_query_master.queries import Query
from train_ticket_auto_query_master.scenarios import *
from common.text_strings import Url_Strings, Admin_API

import json
import requests


def createOrder():

    url = Admin_API.ADMIN_API_URL

    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Admin_API.ADMIN_BEARER_TOKEN}",
    }


    payload = {
        "id": f"{Admin_API.ADMIN_ORDER_ID}",
        "boughtDate": "2026-07-11 01:44:59",
        "travelDate": "2022-10-01 00:00:00",
        "travelTime": "2022-10-01 00:00:00",
        "accountId": f"{Admin_API.ADMIN_USER_ID}",
        "contactsName": "TestEamon",
        "documentType": 1,
        "contactsDocumentNumber": "DocumentNumber_One",
        "trainNumber": "G1237",
        "coachNumber": 5,
        "seatClass": 2,
        "seatNumber": "FirstClass-30",
        "from": "nanjing",
        "to": "shanghaihongqiao",
        "status": 0,
        "price": "100.0",
        "differenceMoney": "0.0",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)       
        print(f"Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")

def deleteOrder():

    url = Url_Strings.BASE_URL
    q = Query(Url_Strings.BASE_URL)


    if q.login():
        url = Admin_API.ADMIN_API_URL
        print(len(q.query_orders()))
        test_order = q.query_orders()[0]  
        ORDER_ID = test_order[0]
        TRAVEL_ID = test_order[1]
        
        delete_url = url + f"/{ORDER_ID}/{TRAVEL_ID}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Admin_API.ADMIN_BEARER_TOKEN}",
        }

        try:
            response = requests.delete(delete_url, headers=headers)      
            print(f"Status Code: {response.status_code}")
            print(f"[OK] Remaining Orders: {len(q.query_orders())}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] An error occurred while making the request:\n{e}")
    else:
        print("[ERROR] Login Failed")
        
        
def execute_TT_Pipeline():

    url = Url_Strings.BASE_URL

    q = Query(url)

    q.login()

    createOrder()
    test_order = q.query_orders()[0]
  
    ORDER_ID = test_order[0]
    TRAVEL_ID = test_order[1]
    print(ORDER_ID, TRAVEL_ID)
    
    q.login()
    time.sleep(3)
    q.pay_order(ORDER_ID, TRAVEL_ID)
    time.sleep(3)
    q.collect_order(ORDER_ID)
    time.sleep(3)
    q.enter_station(ORDER_ID)
    time.sleep(3)
    deleteOrder()

    print("[INFO] Done")


if __name__ == "__main__":
    print("[INFO] Running...")
    url = Url_Strings.BASE_URL

    q = Query(url)
    #execute_TT_Pipeline()
    i = 0
    while i < 1000:
        q.login()
        #createOrder()
        #time.sleep(1)
        #q.query_orders_all_info()
        #time.sleep(1)
        #deleteOrder()
        #time.sleep(1)
        i+=1
    print("[INFO] Done")

        
        