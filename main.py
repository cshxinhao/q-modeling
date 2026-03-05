from src.pipeline import simple_pipeline
from src.scheduler import simple_window_scheduler


class ConfigReader:
    def __init__(self, config_file):
        self.config_file = config_file


if __name__ == "__main__":
    config_reader = ConfigReader("config.json")

    # TODO: Sample Config, needs to be formatted in config file in a more elegant way
    START_YEAR = 2012
    END_YEAR = 2026
    RETRAIN_MONTH = 6
    WINDOW_TYPE = "rolling"
    WINDOW_SIZE = 24
    
    simple_window_scheduler(
        start_year=START_YEAR,
        end_year=END_YEAR,
        retrain_month=RETRAIN_MONTH,
        window_type=WINDOW_TYPE,
        window_size=WINDOW_SIZE,
        pipeline=simple_pipeline,
    )
