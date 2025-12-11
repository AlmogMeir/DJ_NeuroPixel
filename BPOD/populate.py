import pandas as pd
import numpy as np
from scipy.io import loadmat
import sys
import os
# import datajoint as dj

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the tables
# from BPOD.mazeEXP import Port, Block, Reward, LickEvent, schema, exp

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

def extract_reward_info(raw_events_trial, trial_start_time_absolute):
    """Extract detailed reward information from RawEvents.Trial.States structure.
    
    Returns:
        Dictionary with reward information or None if no reward found
    """
    try:
        if hasattr(raw_events_trial, 'States'):
            states = raw_events_trial.States
            
            # Look for RewardX fields (Reward1, Reward2, etc.)
            for attr_name in dir(states):
                if attr_name.startswith('Reward') and not attr_name.endswith('_'):
                    reward_times = getattr(states, attr_name)
                    
                    # Extract port number from Reward name (e.g., Reward1 -> port 1)
                    try:
                        port_number = int(attr_name.replace('Reward', ''))
                    except:
                        port_number = 1  # Default
                    
                    if isinstance(reward_times, (list, np.ndarray)) and len(reward_times) >= 2:
                        # Reward size = second time point - first time point
                        reward_size = float(reward_times[1] - reward_times[0])
                        abs_time = float(reward_times[0])  # Use start time as absolute time
                        
                        return {
                            'port_number': port_number,
                            'reward_size': reward_size,
                            'base_reward_size': reward_size,  # May be modified by depletion
                            'abs_time': abs_time,
                            'depletion_step': 0,  # To be calculated if depletion is implemented
                            'depletion_size': 0.0  # base_reward_size / reward_size if depleted
                        }
                    elif hasattr(reward_times, '__len__') and len(reward_times) >= 2:
                        reward_size = float(reward_times[1] - reward_times[0])
                        abs_time = float(reward_times[0])
                        
                        return {
                            'port_number': port_number,
                            'reward_size': reward_size,
                            'base_reward_size': reward_size,
                            'abs_time': abs_time,
                            'depletion_step': 0,
                            'depletion_size': 0.0
                        }
        
        return None  # No reward found
    except Exception as e:
        print(f"  Error extracting reward info: {e}")
        return None

def populate_behavioral_tables(matlab_file_path, subject_id, session):
    """Main function to populate behavioral tables from MATLAB data.
    
    Args:
        matlab_file_path: Path to the MATLAB .mat file
        subject_id: Subject ID for the session
        session: Session number
    """
    
    # Load MATLAB data
    print(f"Loading MATLAB file: {matlab_file_path}")
    data = load_matlab_data(matlab_file_path)
    
    if data is None:
        print("Failed to load MATLAB data")
        return None, None, None, None
    
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
        return None, None, None, None
    
    session_data = data[main_key]
    print(f"Using main data structure: {main_key}")
    
    # Initialize lists to collect data for each table
    port_data = []
    block_data = []
    reward_data = []
    lick_events_data = []
    
    # Extract port reward sizes from session data
    # This may be stored in session settings or trial data
    port_reward_sizes = {}
    
    # Track which blocks exist (unique combinations of ports)
    blocks_seen = {}
    block_counter = 0
    
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
            
            # Extract lick events from RawEvents.Trial.Events
            lick_events = extract_lick_events(raw_trial, trial_start_time)
            
            # Extract reward information from States
            reward_info = extract_reward_info(raw_trial, trial_start_time)
            
            # Add reward data for this trial if reward was given
            if reward_info:
                reward_entry = {
                    'subject_id': subject_id,
                    'session': session,
                    'trial': trial_num,
                    'port_number': reward_info['port_number'],
                    'reward_size': reward_info['reward_size'],
                    'abs_time': reward_info['abs_time'],
                    'depletion_step': reward_info.get('depletion_step', 0),
                    'depletion_size': reward_info.get('depletion_size', 0.0)
                }
                reward_data.append(reward_entry)
                
                # Track port reward sizes
                port_num = reward_info['port_number']
                if port_num not in port_reward_sizes:
                    port_reward_sizes[port_num] = reward_info.get('base_reward_size', reward_info['reward_size'])
            
            # Add individual lick events - each references the specific trial
            for lick in lick_events:
                # Determine if lick is early (before IRI end)
                is_early = lick.get('is_early', False)
                
                lick_event_entry = {
                    'subject_id': subject_id,
                    'session': session,
                    'trial': trial_num,
                    'lick_id': lick['lick_id'],
                    'port_number': lick['port_number'],
                    'lick_time': lick['lick_time'],
                    'lick_duration': lick['lick_duration'],
                    'lick_start_absolute': lick['lick_start_absolute'],
                    'lick_end_absolute': lick['lick_end_absolute'],
                    'is_early': is_early
                }
                lick_events_data.append(lick_event_entry)
                
                # Track ports from licks
                port_num = lick['port_number']
                if port_num not in port_reward_sizes:
                    port_reward_sizes[port_num] = 0.0  # Default, will be updated if we find reward info
    
    # Create Port data from discovered ports
    for port_num, reward_size in port_reward_sizes.items():
        port_entry = {
            'subject_id': subject_id,
            'session': session,
            'port_number': port_num,
            'port_reward_size': reward_size
        }
        port_data.append(port_entry)
    
    # Create DataFrames
    df_port = pd.DataFrame(port_data)
    df_block = pd.DataFrame(block_data)
    df_reward = pd.DataFrame(reward_data)
    df_lick_events = pd.DataFrame(lick_events_data)
    
    print(f"\nCreated DataFrames:")
    print(f"Port: {len(df_port)} rows")
    print(f"Block: {len(df_block)} rows")
    print(f"Reward: {len(df_reward)} rows")
    print(f"LickEvents: {len(df_lick_events)} rows")
    
    # Display sample data
    if not df_port.empty:
        print("\nPort data:")
        print(df_port)
    
    if not df_reward.empty:
        print("\nSample Reward data:")
        print(df_reward.head())
    
    if not df_lick_events.empty:
        print("\nSample LickEvents data:")
        print(df_lick_events.head())
    
    return df_port, df_block, df_reward, df_lick_events

def save_dataframes_to_csv(df_port, df_block, df_reward, df_lick_events, output_dir="./output"):
    """Save DataFrames to CSV files in the specified directory."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save Port DataFrame
        if not df_port.empty:
            port_path = os.path.join(output_dir, "port.csv")
            df_port.to_csv(port_path, index=False)
            print(f"Saved Port data to: {port_path}")
        
        # Save Block DataFrame
        if not df_block.empty:
            block_path = os.path.join(output_dir, "block.csv")
            df_block.to_csv(block_path, index=False)
            print(f"Saved Block data to: {block_path}")
        
        # Save Reward DataFrame
        if not df_reward.empty:
            reward_path = os.path.join(output_dir, "reward.csv")
            df_reward.to_csv(reward_path, index=False)
            print(f"Saved Reward data to: {reward_path}")
        
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
            f.write(f"Port records: {len(df_port)}\n")
            f.write(f"Block records: {len(df_block)}\n")
            f.write(f"Reward records: {len(df_reward)}\n")
            f.write(f"LickEvents records: {len(df_lick_events)}\n\n")
            
            if not df_port.empty:
                f.write("Port columns:\n")
                for col in df_port.columns:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            if not df_reward.empty:
                f.write("Reward columns:\n")
                for col in df_reward.columns:
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

def insert_data_to_datajoint(df_port, df_block, df_reward, df_lick_events):
    """Insert DataFrames into DataJoint tables."""
    
    try:
        print("\nInserting data into DataJoint tables...")
        
        # Insert Port data first (no foreign key dependencies)
        if not df_port.empty:
            print(f"\nInserting {len(df_port)} Port records...")
            print("Sample Port data:")
            print(df_port.head())
            Port.insert(df_port.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Port records")
        
        # Insert Block data (depends on Port)
        if not df_block.empty:
            print(f"\nInserting {len(df_block)} Block records...")
            print("Sample Block data:")
            print(df_block.head())
            Block.insert(df_block.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Block records")
        
        # Insert Reward data (depends on exp.SessionTrial and Port)
        if not df_reward.empty:
            print(f"\nInserting {len(df_reward)} Reward records...")
            print("Sample Reward data:")
            print(df_reward.head(2))
            Reward.insert(df_reward.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Reward records")
        
        # Insert LickEvent data (depends on exp.SessionTrial and Port)
        if not df_lick_events.empty:
            print(f"\nInserting {len(df_lick_events)} LickEvent records...")
            print("Sample LickEvent data:")
            print(df_lick_events.head(2))
            LickEvent.insert(df_lick_events.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted LickEvent records")
            
        print("\n✓ Data insertion completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error inserting data: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Check that exp.Session records exist for this session")
        print("2. Check that exp.SessionTrial records exist for all trials")
        print("3. Verify column names match table definitions")

if __name__ == "__main__":
    # Configuration
    subject_id = 105101  # Update this with your subject ID
    session = 2  # Update this with your session number
    
    # Path to the MATLAB file
    matlab_file = "NPC1_maze_4pairs_AnA_shany_20250721_120421.mat"
    matlab_path = os.path.join(os.path.dirname(__file__), matlab_file)
    
    # Populate tables
    df_port, df_block, df_reward, df_lick_events = populate_behavioral_tables(
        matlab_path, subject_id, session
    )
    
    if df_port is not None:
        # Save DataFrames to CSV files
        output_directory = os.path.join(os.path.dirname(__file__), "output")
        save_dataframes_to_csv(df_port, df_block, df_reward, df_lick_events, output_directory)
        
        # Uncomment the next line to actually insert data into DataJoint
        # insert_data_to_datajoint(df_port, df_block, df_reward, df_lick_events)
    else:
        print("Failed to process MATLAB data")
