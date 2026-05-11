from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
default_args = {
    'owner': 'mlops_automation_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'politician_cnn_training_pipeline',
    default_args=default_args,
    description='Automated orchestration of CNN training using DVC and MLflow',
    schedule_interval='@weekly',
    catchup=False,
) as dag:

    pull_dvc_data = BashOperator(
        task_id='dvc_pull_latest_data',
        bash_command='dvc pull'
    )

    run_dvc_pipeline = BashOperator(
        task_id='execute_training_and_evaluation',
        bash_command='dvc repro'
    )
    pull_dvc_data >> run_dvc_pipeline