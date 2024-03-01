import psycopg2
from contextlib import contextmanager


@contextmanager
def get_cursor():
    conn = psycopg2.connect(
        host="localhost",
        database="nouveaux_metiers",
        user="oussama",
        password="oussama",
    )
    try:
        cursor = conn.cursor()
        yield cursor
    finally:
        conn.commit()
        conn.close()


with get_cursor() as cursor:
    cursor.execute(
        """
            CREATE TABLE raw_employment_data (
                annee INT,
                BE NUMERIC,
                NOMBE VARCHAR(255),
                Famille_met CHAR(1),
                Lbl_fam_met VARCHAR(255),
                code_metier_BMO VARCHAR(10),
                nommetier VARCHAR(255),
                Dept VARCHAR(10),
                NomDept VARCHAR(255),
                met VARCHAR(100),
                xmet VARCHAR(100), -- Assuming variability and potential non-numeric values
                smet VARCHAR(100), -- Assuming variability and potential non-numeric values
                REG NUMERIC,
                NOM_REG VARCHAR(255)
);

        """
    )
