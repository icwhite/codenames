import pandas as pd

ALL_DEMOS = ['', 'political', 'leaning_only', 'event_only', 'demo_only', 'personality_only', 'all_text']

def add_player_demo_col(demo: str, df: pd.DataFrame, is_giver: bool) -> pd.DataFrame:
    """
    For the provided DataFrame, adds a demographic column providing additional information about
    the clue giver or guesser based on the specified `is_giver`.
    """
    def get_giver_demographic_prompt(s: str):
        s = s.split('GUESSER')[0]
        if demo == 'political':
            s = s.split('political: ')[1].split('[')[0].split(']')[0].strip()
        else:
            s = s.split('[')[1].split(']')[0].strip()
        formatted_s = f'Here is some information about the clue giver: {s}.'
        return formatted_s
    
    def get_guesser_demographic_prompt(s: str):
        # when demographics aren't provided for the guesser, skip over populating that column
        if 'GUESSER: [None]' in s:
            return ''
        
        s = s.split('GUESSER')[1]
        if demo == 'political':
            s = s.split('political: ')[1].split(']')[0].strip()
        else:
            s = s.split('[')[1].split(']')[0].strip()
        formatted_s = f'Here is some information about the clue guesser: {s}.'
        return formatted_s
    
    demo_col = demo if demo != 'political' else 'leaning_only' 
    fn = get_giver_demographic_prompt if is_giver else get_guesser_demographic_prompt
    df['demo_text'] = df[demo_col].apply(fn) if demo else ''
    return df