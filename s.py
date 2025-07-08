import os
import sys
import argparse
import httpx

def get_token(ca_cert: str):
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    token_url = os.getenv('TOKEN_URL')

    if not all([client_id, client_secret, token_url]):
        sys.stderr.write("Error: CLIENT_ID, CLIENT_SECRET, and TOKEN_URL must be set as environment variables.\n")
        sys.exit(1)

    response = httpx.post(
        token_url,
        data={'grant_type': 'client_credentials', 'scope': 'openid'},
        auth=(client_id, client_secret),
        verify=ca_cert or True,
        timeout=10.0
    )
    response.raise_for_status()
    token = response.json().get('access_token')
    if not token:
        sys.stderr.write("Error: No access_token in token response.\n")
        sys.exit(1)
    return token

def toggle_dag(dag_id: str, pause: bool, ca_cert: str):
    airflow_host = os.getenv('AIRFLOW_HOST')
    if not airflow_host:
        sys.stderr.write("Error: AIRFLOW_HOST must be set as an environment variable.\n")
        sys.exit(1)

    token = get_token(ca_cert)
    url = f"{airflow_host.rstrip('/')}/api/v1/dags/{dag_id}"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    body = {'is_paused': pause}

    response = httpx.patch(
        url,
        headers=headers,
        json=body,
        verify=ca_cert or True,
        timeout=10.0
    )
    response.raise_for_status()
    action = "paused" if pause else "unpaused"
    print(f"DAG '{dag_id}' successfully {action}.")

def main():
    parser = argparse.ArgumentParser(description="Pause or unpause an Airflow DAG via the REST API using httpx.")
    parser.add_argument('dag_id', help="The ID of the DAG to toggle (e.g., all_I-MAD).")
    parser.add_argument('action', choices=['pause', 'unpause'], help="Whether to pause or unpause the DAG.")
    parser.add_argument('--ca-cert', dest='ca_cert',
                        help="Path to CA certificate PEM file for SSL verification",
                        default=None)
    args = parser.parse_args()

    toggle_dag(args.dag_id, pause=(args.action == 'pause'), ca_cert=args.ca_cert)

if __name__ == "__main__":
    main()

