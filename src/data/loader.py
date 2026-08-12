import fastf1 as f1
import os 

def load_session(year: int, race: int, session_type: str = 'R'):
    os.makedirs('cache', exist_ok=True)
    f1.Cache.enable_cache('cache')
    session = f1.get_session(year,race,session_type)
    session.load(telemetry=False, messages=False)

    return session

def get_weather(session) -> dict:
    return{
        'tracktemp':session.weather_data['TrackTemp'].mean(),
        'humidity':session.weather_data['Humidity'].mean(),
        'rainfall':session.weather_data['Rainfall'].any()
    }