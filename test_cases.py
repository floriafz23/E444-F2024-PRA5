import requests
import pytest
import time
import csv

url = "http://serve-sentiment-env.eba-s8hsx6mm.us-east-2.elasticbeanstalk.com/prediction"
test_cases = [
    {"input": "real news"},
    {"input": "fake news"},
    {"input": "another real news"},
    {"input": "another fake news"}
]

expected_results = {
    "real news": "REAL",
    "fake news": "FAKE",
    "another real news": "REAL",
    "another fake news": "FAKE"
}


@pytest.mark.parametrize("test_case", test_cases)
def test_model_accuracy(test_case):
    response = requests.post(url, json=test_case)
    assert response.status_code == 200
    assert response.text == expected_results[test_case["input"]]

def test_latency():
    latencies = []
    for _ in range(100):
        start_time = time.time()
        response = requests.post(url, json={"input": "real news piece 1"})
        end_time = time.time()
        latencies.append(end_time - start_time)
        assert response.status_code == 200

    with open('latency_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["No", "Latency (s)"])
        for i, latency in enumerate(latencies, start=1):
            writer.writerow([i, latency])