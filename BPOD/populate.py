import pandas as pd
import numpy as np
from scipy.io import loadmat
import sys
import os
import datajoint as dj

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the tables
from BPOD.mazeEXP import TrialBehavior, LickEvent, schema, exp

EXPt = dj.VirtualModule("EXPt", "talch012_expt", create_tables=True)
session_key = (EXPt.Session & 'session=2' & 'subject_id=105101').fetch1('session', 'subject_id')

def load_matlab_data(file_path):
    """Load MATLAB file and return the data structure."""
    try:
        data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        return data
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None

def extract_lick_events(raw_events_trial, trial_start_time_absolute):
    """Extract lick events from RawEvents.Trial.Events structure.
    Pairs PortXIn and PortXOut to calculate lick duration.
    
    Args:
        raw_events_trial: The trial data from RawEvents.Trial
        trial_start_time_absolute: Absolute trial start time for reference
    """
    lick_events = []
    lick_id = 0
    
    if hasattr(raw_events_trial, 'Events'):
        events = raw_events_trial.Events
        
        # First, collect all port events by port number
        port_events = {}  # {port_number: {'In': [...], 'Out': [...]}}
        
        for attr_name in dir(events):
            if attr_name.startswith('Port') and ('In' in attr_name or 'Out' in attr_name):
                # Extract port number and event type
                if 'In' in attr_name:
                    port_num = int(attr_name.replace('Port', '').replace('In', ''))
                    event_type = 'In'
                elif 'Out' in attr_name:
                    port_num = int(attr_name.replace('Port', '').replace('Out', ''))
                    event_type = 'Out'
                
                # Initialize port if not exists
                if port_num not in port_events:
                    port_events[port_num] = {'In': [], 'Out': []}
                
                # Get the event times
                event_times = getattr(events, attr_name)
                if isinstance(event_times, (list, np.ndarray)):
                    port_events[port_num][event_type] = list(event_times)
                elif isinstance(event_times, (int, float)):
                    port_events[port_num][event_type] = [event_times]
        
        # Now pair In and Out events for each port
        for port_num, events_dict in port_events.items():
            in_times = events_dict['In']
            out_times = events_dict['Out']
            
            # Ensure both lists are sorted
            in_times.sort()
            out_times.sort()
            
            # Pair In and Out events (assume they correspond by array position)
            max_pairs = min(len(in_times), len(out_times))
            
            for i in range(max_pairs):
                lick_start_time_raw = in_times[i]
                lick_end_time_raw = out_times[i]
                
                # Calculate lick duration (this should work regardless of absolute vs relative)
                lick_duration = float(lick_end_time_raw - lick_start_time_raw)
                
                # Determine if lick times are absolute or relative
                # If lick times are much smaller than trial start time, they're likely relative
                if lick_start_time_raw < trial_start_time_absolute:
                    # Lick times appear to be relative to trial start
                    lick_time_relative = float(lick_start_time_raw)
                    lick_start_absolute = float(trial_start_time_absolute + lick_start_time_raw)
                    lick_end_absolute = float(trial_start_time_absolute + lick_end_time_raw)
                else:
                    # Lick times appear to be absolute
                    lick_time_relative = float(lick_start_time_raw - trial_start_time_absolute)
                    lick_start_absolute = float(lick_start_time_raw)
                    lick_end_absolute = float(lick_end_time_raw)
                
                # Ensure duration is positive (sanity check)
                if lick_duration >= 0:
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': lick_time_relative,  # Relative to trial start
                        'lick_duration': lick_duration,  # Duration of the lick
                        'lick_start_absolute': lick_start_absolute,  # Absolute start time
                        'lick_end_absolute': lick_end_absolute  # Absolute end time
                    })
                    lick_id += 1
            
            # Handle unpaired events (more In than Out or vice versa)
            if len(in_times) > len(out_times):
                # More In events than Out events
                for i in range(max_pairs, len(in_times)):
                    lick_start_time_raw = in_times[i]
                    
                    # Determine if relative or absolute
                    if lick_start_time_raw < trial_start_time_absolute:
                        lick_time_relative = float(lick_start_time_raw)
                        lick_start_absolute = float(trial_start_time_absolute + lick_start_time_raw)
                    else:
                        lick_time_relative = float(lick_start_time_raw - trial_start_time_absolute)
                        lick_start_absolute = float(lick_start_time_raw)
                    
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': lick_time_relative,
                        'lick_duration': 0.0,  # Unknown duration
                        'lick_start_absolute': lick_start_absolute,
                        'lick_end_absolute': None  # No corresponding Out event
                    })
                    lick_id += 1
            elif len(out_times) > len(in_times):
                # More Out events than In events (unusual, but handle it)
                for i in range(max_pairs, len(out_times)):
                    lick_end_time_raw = out_times[i]
                    
                    # Determine if relative or absolute
                    if lick_end_time_raw < trial_start_time_absolute:
                        lick_time_relative = float(lick_end_time_raw)
                        lick_end_absolute = float(trial_start_time_absolute + lick_end_time_raw)
                    else:
                        lick_time_relative = float(lick_end_time_raw - trial_start_time_absolute)
                        lick_end_absolute = float(lick_end_time_raw)
                    
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': lick_time_relative,
                        'lick_duration': 0.0,  # Unknown duration
                        'lick_start_absolute': None,  # No corresponding In event
                        'lick_end_absolute': lick_end_absolute
                    })
                    lick_id += 1
    
    return lick_events

def extract_trial_times(raw_events_trial):
    """Extract trial start and end times from RawEvents.Trial.States structure."""
    trial_start_time = None
    trial_end_time = None
    
    try:
        if hasattr(raw_events_trial, 'States'):
            states = raw_events_trial.States
            
            # Look for TrialStart
            if hasattr(states, 'TrialStart'):
                trial_start = states.TrialStart
                if isinstance(trial_start, (list, np.ndarray)):
                    trial_start_time = float(trial_start[0]) if len(trial_start) > 0 else None
                else:
                    trial_start_time = float(trial_start)
            
            # Look for TrialEnd
            if hasattr(states, 'TrialEnd'):
                trial_end = states.TrialEnd
                if isinstance(trial_end, (list, np.ndarray)):
                    trial_end_time = float(trial_end[-1]) if len(trial_end) > 0 else None
                else:
                    trial_end_time = float(trial_end)
    except:
        pass
    
    return trial_start_time, trial_end_time

def calculate_reward_size(raw_events_trial):
    """Calculate reward size from RawEvents.Trial.States structure."""
    try:
        if hasattr(raw_events_trial, 'States'):
            states = raw_events_trial.States
            # Look for RewardX fields (Reward1, Reward2, etc.)
            for attr_name in dir(states):
                if attr_name.startswith('Reward'):
                    reward_times = getattr(states, attr_name)
                    if isinstance(reward_times, (list, np.ndarray)) and len(reward_times) >= 2:
                        # Reward size = second time point - first time point
                        reward_duration = float(reward_times[1] - reward_times[0])
                        print(f"  Found reward: {attr_name}, duration = {reward_duration:.3f}s")
                        return reward_duration
                    elif hasattr(reward_times, '__len__') and len(reward_times) >= 2:
                        reward_duration = float(reward_times[1] - reward_times[0])
                        print(f"  Found reward: {attr_name}, duration = {reward_duration:.3f}s")
                        return reward_duration
        
        print("  No reward found, using default 0.0")
        return 0.0  # Default value if no reward found
    except Exception as e:
        print(f"  Error calculating reward size: {e}, using default 0.0")
        return 0.0

def populate_behavioral_tables(matlab_file_path):
    """Main function to populate behavioral tables from MATLAB data."""
    
    # Load MATLAB data
    print(f"Loading MATLAB file: {matlab_file_path}")
    data = load_matlab_data(matlab_file_path)
    
    if data is None:
        print("Failed to load MATLAB data")
        return
    
    # Print available keys to understand structure
    print("Available keys in MATLAB data:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key])}")
    
    # Find the main data structure (often the first non-underscore key)
    main_key = None
    for key in data.keys():
        if not key.startswith('__'):
            main_key = key
            break
    
    if main_key is None:
        print("No main data structure found")
        return
    
    session_data = data[main_key]
    print(f"Using main data structure: {main_key}")
    
    # Initialize lists to collect ALL trial data
    trial_behavior_data = []
    lick_events_data = []
    
    # Extract session info (you may need to adjust these based on actual structure)
    try:
        subject_id = getattr(session_data, 'subject_id', 105101)  # Default value
        session_date = getattr(session_data, 'session_date', '2025-08-05')  # Default
    except:
        subject_id = 105101
        session_date = '2025-08-05'
    
    # Process trials - get trial data from RawEvents.Trial structure
    if hasattr(session_data, 'RawEvents') and hasattr(session_data.RawEvents, 'Trial'):
        raw_trials = session_data.RawEvents.Trial
        if not isinstance(raw_trials, (list, np.ndarray)):
            raw_trials = [raw_trials]  # Single trial case
        
        # Get absolute trial timestamps from SessionData
        trial_start_timestamps = None
        trial_end_timestamps = None
        
        if hasattr(session_data, 'TrialStartTimestamp'):
            trial_start_timestamps = session_data.TrialStartTimestamp
            if not isinstance(trial_start_timestamps, (list, np.ndarray)):
                trial_start_timestamps = [trial_start_timestamps]
        
        if hasattr(session_data, 'TrialEndTimestamp'):
            trial_end_timestamps = session_data.TrialEndTimestamp
            if not isinstance(trial_end_timestamps, (list, np.ndarray)):
                trial_end_timestamps = [trial_end_timestamps]
        
        print(f"Found {len(raw_trials)} trials")
        if trial_start_timestamps is not None and len(trial_start_timestamps) > 0:
            print(f"Found {len(trial_start_timestamps)} trial start timestamps")
        if trial_end_timestamps is not None and len(trial_end_timestamps) > 0:
            print(f"Found {len(trial_end_timestamps)} trial end timestamps")
        
        for trial_idx, raw_trial in enumerate(raw_trials):
            trial_num = trial_idx + 1
            
            # Get absolute trial times from SessionData timestamps
            trial_start_time_absolute = None
            trial_end_time_absolute = None
            
            if trial_start_timestamps is not None and len(trial_start_timestamps) > trial_idx:
                trial_start_time_absolute = float(trial_start_timestamps[trial_idx])
            
            if trial_end_timestamps is not None and len(trial_end_timestamps) > trial_idx:
                trial_end_time_absolute = float(trial_end_timestamps[trial_idx])
            
            # Extract relative trial start and end times from States (if available)
            trial_start_time_relative, trial_end_time_relative = extract_trial_times(raw_trial)
            
            # Use absolute times if available, otherwise fall back to relative or defaults
            if trial_start_time_absolute is not None:
                trial_start_time = trial_start_time_absolute
            elif trial_start_time_relative is not None:
                trial_start_time = trial_start_time_relative
            else:
                trial_start_time = float(trial_idx * 10)  # Default spacing
            
            if trial_end_time_absolute is not None:
                trial_end_time = trial_end_time_absolute
            elif trial_end_time_relative is not None:
                trial_end_time = trial_end_time_relative
            else:
                trial_end_time = trial_start_time + 10.0  # Default 10 seconds
            
            # print(f"Trial {trial_num}: Start={trial_start_time:.3f}s, End={trial_end_time:.3f}s")
            
            # Extract blocks information
            blocks = getattr(session_data, 'Blocks', [])  # Adjust based on actual structure
            if isinstance(blocks, (list, np.ndarray)) and len(blocks) > trial_idx:
                trial_blocks = blocks[trial_idx] if len(blocks) > trial_idx else []
            else:
                trial_blocks = []
            
            # Extract lick events from RawEvents.Trial.Events
            lick_events = extract_lick_events(raw_trial, trial_start_time)
            
            # Calculate reward size from RawEvents.Trial.States
            reward_size = calculate_reward_size(raw_trial)
            
            # Ensure reward_size is never None/null
            if reward_size is None or np.isnan(reward_size):
                reward_size = 0.0
            
            print(f"Trial {trial_num}: Start={trial_start_time:.3f}s, End={trial_end_time:.3f}s, Reward={reward_size:.3f}s")
            
            # Create trial behavior entry - one record per trial with trial as primary key component
            trial_behavior_entry = {
                'session': 2,
                'subject_id': subject_id,
                'trial': trial_num,  # Now part of primary key: (subject_id, session, trial)
                'trial_start_time': float(trial_start_time),
                'trial_end_time': float(trial_end_time),
                'reward_size': reward_size,
                'licks': lick_events,  # Store licks for THIS trial
                'blocks': trial_blocks  # Store blocks for THIS trial
            }
            trial_behavior_data.append(trial_behavior_entry)
            
            # Add individual lick events - each references the specific trial
            for lick in lick_events:
                lick_event_entry = {
                    'session': 2,
                    'subject_id': subject_id,
                    'trial': trial_num,  # Now part of foreign key to TrialBehavior
                    'lick_id': lick['lick_id'],  # Can be unique within trial
                    'port_number': lick['port_number'],
                    'lick_time': lick['lick_time'],
                    'lick_duration': lick['lick_duration'],
                    'lick_start_absolute': lick['lick_start_absolute'],
                    'lick_end_absolute': lick['lick_end_absolute']
                }
                lick_events_data.append(lick_event_entry)
    
    # Create DataFrames
    df_trial_behavior = pd.DataFrame(trial_behavior_data)
    df_lick_events = pd.DataFrame(lick_events_data)
    
    print(f"\nCreated DataFrames:")
    print(f"TrialBehavior: {len(df_trial_behavior)} rows")
    print(f"LickEvents: {len(df_lick_events)} rows")
    
    # Display sample data
    if not df_trial_behavior.empty:
        print("\nSample TrialBehavior data:")
        print(df_trial_behavior.head())
    
    if not df_lick_events.empty:
        print("\nSample LickEvents data:")
        print(df_lick_events.head())
    
    return df_trial_behavior, df_lick_events

def save_dataframes_to_csv(df_trial_behavior, df_lick_events, output_dir="./output"):
    """Save DataFrames to CSV files in the specified directory."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save TrialBehavior DataFrame
        if not df_trial_behavior.empty:
            trial_behavior_path = os.path.join(output_dir, "trial_behavior.csv")
            df_trial_behavior.to_csv(trial_behavior_path, index=False)
            print(f"Saved TrialBehavior data to: {trial_behavior_path}")
        
        # Save LickEvents DataFrame
        if not df_lick_events.empty:
            lick_events_path = os.path.join(output_dir, "lick_events.csv")
            df_lick_events.to_csv(lick_events_path, index=False)
            print(f"Saved LickEvents data to: {lick_events_path}")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, "data_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Data Processing Summary\n")
            f.write("=====================\n\n")
            f.write(f"TrialBehavior records: {len(df_trial_behavior)}\n")
            f.write(f"LickEvents records: {len(df_lick_events)}\n\n")
            
            if not df_trial_behavior.empty:
                f.write("TrialBehavior columns:\n")
                for col in df_trial_behavior.columns:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            if not df_lick_events.empty:
                f.write("LickEvents columns:\n")
                for col in df_lick_events.columns:
                    f.write(f"  - {col}\n")
        
        print(f"Saved data summary to: {summary_path}")
        print(f"\nAll files saved to directory: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"Error saving CSV files: {e}")

    return df_trial_behavior, df_lick_events

def insert_data_to_datajoint(df_trial_behavior, df_lick_events):
    """Insert DataFrames into DataJoint tables."""
    
    try:
        # First, check what's in the parent table to understand the foreign key structure
        print("Checking foreign key constraints...")
        
        # Query the parent table to see what sessions exist
        try:
            existing_sessions = (exp.Session).fetch(as_dict=True)
            print(f"Found {len(existing_sessions)} existing sessions in exp.Session")
            if len(existing_sessions) > 0:
                print("Sample existing session record:")
                print(existing_sessions[0])
                print("Available columns:", list(existing_sessions[0].keys()))
        except Exception as e:
            print(f"Could not query existing sessions: {e}")
        
        # Updated column structure for corrected table definitions
        # TrialBehavior primary key: (subject_id, session, trial) where trial is now part of PK
        trial_behavior_columns = ['subject_id', 'session', 'trial',  # Complete primary key
                                'trial_start_time', 'trial_end_time', 'reward_size', 'licks', 'blocks']
        
        # LickEvent references TrialBehavior with full primary key
        lick_event_columns = ['subject_id', 'session', 'trial', 'lick_id',
                             'port_number', 'lick_time', 'lick_duration', 
                             'lick_start_absolute', 'lick_end_absolute']
        
        # Filter DataFrames to only include relevant columns
        if not df_trial_behavior.empty:
            print("Original TrialBehavior columns:", list(df_trial_behavior.columns))
            
            # Check if we need to rename session_time to session or create session column
            df_trial_filtered = df_trial_behavior.copy()
            
            # If session column doesn't exist but session_time does, we may need to map it
            if 'session' not in df_trial_filtered.columns and 'session_time' in df_trial_filtered.columns:
                # You may need to adjust this mapping based on your data
                print("Warning: 'session' column not found, may need to create it from existing data")
                # Option 1: Use a fixed session number
                df_trial_filtered['session'] = 1  # or whatever session number you're using
                # Option 2: Extract session from session_time or other logic
                # df_trial_filtered['session'] = extract_session_number(df_trial_filtered['session_time'])
            
            # Filter to only needed columns that exist
            available_trial_cols = [col for col in trial_behavior_columns if col in df_trial_filtered.columns]
            df_trial_filtered = df_trial_filtered[available_trial_cols]
            
            print(f"Filtered TrialBehavior columns: {list(df_trial_filtered.columns)}")
            print("Sample data to be inserted:")
            print(df_trial_filtered.head(2))
            
            # Before inserting, check if the foreign key records exist
            sample_record = df_trial_filtered.iloc[0]
            foreign_key_check = {'subject_id': sample_record['subject_id'], 
                                'session': sample_record['session']}  # Only session and subject_id for exp.Session
            
            try:
                existing_record = (exp.Session & foreign_key_check).fetch()
                if len(existing_record) == 0:
                    print(f"WARNING: No matching record found in exp.Session for: {foreign_key_check}")
                    print("You may need to populate the exp.Session table first.")
                    return
                else:
                    print(f"Foreign key validation passed for sample record: {foreign_key_check}")
            except Exception as e:
                print(f"Could not validate foreign key: {e}")
            
            print("Inserting TrialBehavior data...")
            
            # Debug: Check for any problematic values before insertion
            print("Checking data quality before insertion...")
            for i, record in enumerate(df_trial_filtered.to_dict('records')):
                if record.get('reward_size') is None or pd.isna(record.get('reward_size')):
                    print(f"WARNING: Record {i} has invalid reward_size: {record.get('reward_size')}")
                    record['reward_size'] = 0.0  # Fix it
                # print(f"Record {i}: reward_size = {record['reward_size']}")
            
            TrialBehavior.insert(df_trial_filtered.to_dict('records'))
            print(f"Inserted {len(df_trial_filtered)} TrialBehavior records")
        
        # Similar process for LickEvent data
        if not df_lick_events.empty:
            print("Original LickEvent columns:", list(df_lick_events.columns))
            
            df_lick_filtered = df_lick_events.copy()
            
            # Handle session column mapping for lick events too
            if 'session' not in df_lick_filtered.columns and 'session_time' in df_lick_filtered.columns:
                df_lick_filtered['session'] = 1  # Match the session used for trials
            
            # Filter to only needed columns that exist
            available_lick_cols = [col for col in lick_event_columns if col in df_lick_filtered.columns]
            df_lick_filtered = df_lick_filtered[available_lick_cols]
            
            print(f"Filtered LickEvent columns: {list(df_lick_filtered.columns)}")
            
            print("Inserting LickEvent data...")
            LickEvent.insert(df_lick_filtered.to_dict('records'))
            print(f"Inserted {len(df_lick_filtered)} LickEvent records")
            
        print("Data insertion completed successfully!")
        
    except Exception as e:
        print(f"Error inserting data: {e}")
        print("This might be due to missing foreign key records or column mismatch.")
        print("Check that the exp.Session records exist for these sessions.")

if __name__ == "__main__":
    # Path to the MATLAB file
    matlab_file = "NPC1_maze_4pairs_AnA_shany_20250721_120421.mat"
    matlab_path = os.path.join(os.path.dirname(__file__), matlab_file)
    
    # Populate tables
    df_trial_behavior, df_lick_events = populate_behavioral_tables(matlab_path)
    
    # Save DataFrames to CSV files
    output_directory = os.path.join(os.path.dirname(__file__), "output")
    save_dataframes_to_csv(df_trial_behavior, df_lick_events, output_directory)
    
    # Uncomment the next line to actually insert data into DataJoint
    insert_data_to_datajoint(df_trial_behavior, df_lick_events)
