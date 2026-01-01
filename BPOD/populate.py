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

# Import the tables - COMMENT OUT TO AVOID DJ CONNECTION
from BPOD.mazeBPOD import Port, Block, Reward, LickEvent, SessionTrial, schema, exp

if not all([Port, Block, Reward, LickEvent, SessionTrial]):
    raise ImportError("Failed to import required tables from BPOD.mazeBPOD")

def load_matlab_data(file_path):
    """Load MATLAB file and return the data structure."""
    try:
        data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        return data
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None

def extract_lick_events(raw_events_trial, iri_end_time=None):
    """Extract lick events from RawEvents.Trial.Events structure.
    Pairs PortXIn and PortXOut to calculate lick duration.
    Handles multiple ports per trial and marks early licks based on IRI.
    
    All times in Events are RELATIVE to trial start.
    Absolute times will be calculated later using TrialStartTimestamp.
    
    Args:
        raw_events_trial: The trial data from RawEvents.Trial
        iri_end_time: IRI end time (relative to trial start) for early lick detection
    
    Returns:
        List of lick event dictionaries with relative times
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
                
                # Get the event times and filter out NaN values
                event_times = getattr(events, attr_name)
                valid_times = []
                
                if isinstance(event_times, (list, np.ndarray)):
                    # Filter out NaN values
                    event_times_array = np.array(event_times)
                    valid_times = event_times_array[~np.isnan(event_times_array)].tolist()
                elif isinstance(event_times, (int, float)) and not np.isnan(event_times):
                    valid_times = [event_times]
                
                port_events[port_num][event_type] = valid_times
        
        # Now pair In and Out events for each port
        for port_num, events_dict in port_events.items():
            in_times = events_dict['In']
            out_times = events_dict['Out']
            
            # Skip if no events for this port
            if len(in_times) == 0 and len(out_times) == 0:
                continue
            
            # Ensure both lists are sorted
            in_times.sort()
            out_times.sort()
            
            # Handle noise cases in the data
            # Case 1: More Out than In by 1 - drop first Out (false trigger)
            if len(out_times) == len(in_times) + 1:
                out_times = out_times[1:]  # Drop first Out event
            
            # Case 2: Same length but first Out is before first In - drop first Out and last In
            elif len(out_times) == len(in_times) and len(out_times) > 0 and len(in_times) > 0:
                if out_times[0] < in_times[0]:
                    out_times = out_times[1:]  # Drop first Out
                    in_times = in_times[:-1]   # Drop last In
            
            # Pair In and Out events (assume they correspond by array position)
            max_pairs = min(len(in_times), len(out_times))
            
            for i in range(max_pairs):
                lick_start_time_relative = in_times[i]
                lick_end_time_relative = out_times[i]
                
                # Calculate lick duration from relative times
                lick_duration = float(round(lick_end_time_relative - lick_start_time_relative, 4))
                
                # Ensure duration is positive (sanity check)
                if lick_duration >= 0:
                    # Determine if lick is early (before IRI end)
                    is_early = False
                    if iri_end_time is not None:
                        is_early = lick_start_time_relative < iri_end_time
                    
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': float(lick_start_time_relative),  # Relative to trial start
                        'lick_end_time': float(lick_end_time_relative),  # Relative to trial start
                        'lick_duration': lick_duration,
                        'is_early': is_early
                    })
                    lick_id += 1
            
            # Handle unpaired events (more In than Out or vice versa)
            if len(in_times) > len(out_times):
                # More In events than Out events
                for i in range(max_pairs, len(in_times)):
                    lick_start_time_relative = in_times[i]
                    
                    # Determine if lick is early (before IRI end)
                    is_early = False
                    if iri_end_time is not None:
                        is_early = lick_start_time_relative < iri_end_time
                    
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': float(lick_start_time_relative),
                        'lick_end_time': None,  # No corresponding Out event
                        'lick_duration': 0.0,  # Unknown duration
                        'is_early': is_early
                    })
                    lick_id += 1
            elif len(out_times) > len(in_times):
                # More Out events than In events (unusual, but handle it)
                for i in range(max_pairs, len(out_times)):
                    lick_end_time_relative = out_times[i]
                    
                    # Determine if lick is early (before IRI end)
                    is_early = False
                    if iri_end_time is not None:
                        is_early = lick_end_time_relative < iri_end_time
                    
                    lick_events.append({
                        'lick_id': lick_id,
                        'port_number': port_num,
                        'lick_time': float(lick_end_time_relative),
                        'lick_end_time': None,  # No corresponding In event
                        'lick_duration': 0.0,  # Unknown duration
                        'is_early': is_early
                    })
                    lick_id += 1
    
    return lick_events

def extract_trial_times(raw_events_trial):
    """Extract trial start and end times from RawEvents.Trial.States structure."""
    trial_start_time = None
    trial_end_time = None
    iri_end_time = None
    
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
            
            # Look for IRI end time (second time point)
            if hasattr(states, 'IRI'):
                iri = states.IRI
                if isinstance(iri, (list, np.ndarray)):
                    # IRI end time is the second value (index 1)
                    iri_end_time = float(iri[1]) if len(iri) > 1 else None
                else:
                    iri_end_time = float(iri)
    except:
        pass
    
    return trial_start_time, trial_end_time, iri_end_time

def calculate_reward_size(raw_events_trial):
    """Calculate reward size from RawEvents.Trial.States structure.
    
    Reward size is calculated as the difference between RewardXclose and RewardX
    time points for the port that has non-NaN values.
    """
    try:
        if hasattr(raw_events_trial, 'States'):
            states = raw_events_trial.States
            
            # Find reward port (X) that has non-NaN values
            reward_ports = {}
            for attr_name in dir(states):
                if attr_name.startswith('Reward'):
                    reward_times = getattr(states, attr_name)
                    
                    # Extract port number and type (RewardX vs RewardXclose)
                    if 'close' in attr_name.lower():
                        port_num = int(attr_name.replace('Reward', '').replace('close', '').replace('Close', ''))
                        reward_type = 'close'
                    else:
                        port_num = int(attr_name.replace('Reward', ''))
                        reward_type = 'open'
                    
                    # Check if times are valid (not NaN)
                    if isinstance(reward_times, (list, np.ndarray)):
                        reward_times = np.array(reward_times)
                        valid_times = reward_times[~np.isnan(reward_times)]
                        if len(valid_times) > 0:
                            if port_num not in reward_ports:
                                reward_ports[port_num] = {}
                            reward_ports[port_num][reward_type] = float(valid_times[0])
            
            # Calculate reward size for ports that have both open and close times
            for port_num, times_dict in reward_ports.items():
                if 'open' in times_dict and 'close' in times_dict:
                    reward_size = float(times_dict['close'] - times_dict['open'])
                    print(f"  Found reward at port {port_num}: size = {reward_size:.3f}s")
                    return reward_size, port_num
        
        print("  No reward found, using default 0.0")
        return 0.0, None
    except Exception as e:
        print(f"  Error calculating reward size: {e}, using default 0.0")
        return 0.0, None

def extract_reward_info(raw_events_trial, trial_start_time_absolute):
    """Extract detailed reward information from RawEvents.Trial.States structure.
    
    Returns:
        Dictionary with reward information or None if no reward found
    """
    try:
        if hasattr(raw_events_trial, 'States'):
            states = raw_events_trial.States
            
            # Look for RewardX fields (Reward1, Reward2, etc.) that have valid (non-NaN) values
            for attr_name in dir(states):
                if attr_name.startswith('Reward') and not attr_name.endswith('_') and 'close' not in attr_name.lower():
                    reward_times = getattr(states, attr_name)
                    
                    # Check if reward_times contains valid (non-NaN) values
                    if isinstance(reward_times, (list, np.ndarray)):
                        reward_times_array = np.array(reward_times)
                        # Skip if all values are NaN
                        if np.all(np.isnan(reward_times_array)):
                            continue
                        # Filter out NaN values
                        reward_times = reward_times_array[~np.isnan(reward_times_array)]
                    elif isinstance(reward_times, (int, float)) and np.isnan(reward_times):
                        continue
                    
                    # Extract port number from Reward name (e.g., Reward1 -> port 1)
                    try:
                        port_number = int(attr_name.replace('Reward', ''))
                    except:
                        continue  # Skip if we can't extract port number
                    
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
    
    # Extract port reward sizes from SessionData.AllRewardSizes
    port_reward_sizes = {}
    
    if hasattr(session_data, 'AllRewardSizes'):
        all_reward_sizes = session_data.AllRewardSizes
        if not isinstance(all_reward_sizes, (list, np.ndarray)):
            all_reward_sizes = [all_reward_sizes]
        
        # Port numbers start from 1, array index starts from 0
        for idx, reward_size in enumerate(all_reward_sizes):
            port_number = idx + 1  # Port 1, Port 2, Port 3, etc.
            port_reward_sizes[port_number] = float(reward_size)
        
        print(f"Found {len(port_reward_sizes)} ports from AllRewardSizes:")
        for port_num, reward_size in port_reward_sizes.items():
            print(f"  Port {port_num}: {reward_size}s reward")
    else:
        print("Warning: AllRewardSizes not found in session data")
    
    # Extract block information from SessionData.Blocks
    if hasattr(session_data, 'Blocks'):
        blocks = session_data.Blocks
        
        # Convert to list if single block
        if not isinstance(blocks, (list, np.ndarray)):
            blocks = [blocks]
        
        # Get unique block configurations
        unique_blocks = []
        seen_block_configs = set()
        
        for block_config in blocks:
            # Each block_config should be an array of port numbers
            if isinstance(block_config, (list, np.ndarray)):
                block_tuple = tuple(sorted(block_config))  # Sort for consistency
            elif isinstance(block_config, (int, float)):
                block_tuple = (int(block_config),)
            else:
                continue
            
            # Only add if we haven't seen this configuration
            if block_tuple not in seen_block_configs:
                seen_block_configs.add(block_tuple)
                unique_blocks.append(block_tuple)
        
        print(f"Found {len(unique_blocks)} unique block configurations:")
        for block_id, ports_in_block in enumerate(unique_blocks, start=1):
            print(f"  Block {block_id}: Ports {list(ports_in_block)}")
            
            # Create one entry for each port in this block
            for port_number in ports_in_block:
                block_entry = {
                    'subject_id': subject_id,
                    'session': session,
                    'block_id': block_id,
                    'port_number': int(port_number)
                }
                block_data.append(block_entry)
    else:
        print("Warning: SessionData.Blocks not found - no block data will be extracted")
    
    # Extract depletion steps from SessionData.DepletionSteps
    depletion_steps = None
    if hasattr(session_data, 'DepletionStep'):
        depletion_steps = session_data.DepletionStep
        if not isinstance(depletion_steps, (list, np.ndarray)):
            depletion_steps = [depletion_steps]
        print(f"Found {len(depletion_steps)} depletion step values")
    else:
        print("Warning: SessionData.DepletionStep not found")
    
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
            trial_start_time_relative, trial_end_time_relative, iri_end_time = extract_trial_times(raw_trial)
            
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
            
            # Extract lick events from RawEvents.Trial.Events (all times are relative)
            lick_events = extract_lick_events(raw_trial, iri_end_time)
            
            # Extract reward information from States
            reward_info = extract_reward_info(raw_trial, trial_start_time_absolute)
            
            # Add reward data for this trial if reward was given
            if reward_info:
                # Get depletion_step for this trial
                depletion_step = 0
                if depletion_steps is not None and len(depletion_steps) > trial_idx:
                    depletion_step = int(depletion_steps[trial_idx])
                
                # Calculate depletion_size = current_reward_size / base_reward_size
                port_number = reward_info['port_number']
                current_reward_size = reward_info['reward_size']
                
                # Round reward size to 4 decimal places, set very small values to 0
                current_reward_size = round(current_reward_size, 3)
                if abs(current_reward_size) < 0.0001:  # Values smaller than 0.0001 become 0
                    current_reward_size = 0.0
                
                # Get the base reward size from port_reward_sizes
                if port_number in port_reward_sizes:
                    base_reward_size = port_reward_sizes[port_number]
                    # Calculate depletion ratio: current / base
                    # Only calculate if base_reward_size > 0 and current_reward_size > 0
                    if base_reward_size > 0 and current_reward_size > 0:
                        depletion_size = current_reward_size / base_reward_size
                    else:
                        depletion_size = 0.0
                else:
                    # If port not in port_reward_sizes, use the current reward size as base
                    base_reward_size = current_reward_size
                    depletion_size = 1.0 if current_reward_size > 0 else 0.0
                
                # Calculate absolute reward time = relative time + trial start timestamp
                reward_abs_time = reward_info['abs_time']
                if trial_start_timestamps is not None and len(trial_start_timestamps) > trial_idx:
                    reward_abs_time = reward_info['abs_time'] + trial_start_timestamps[trial_idx]
                
                reward_entry = {
                    'subject_id': subject_id,
                    'session': session,
                    'trial': trial_num,
                    'port_number': port_number,
                    'reward_size': current_reward_size,
                    'abs_time': reward_abs_time,
                    'depletion_step': depletion_step,
                    'depletion_size': depletion_size
                }
                reward_data.append(reward_entry)
            
            # Add individual lick events - each references the specific trial
            # Calculate absolute times using TrialStartTimestamp
            for lick in lick_events:
                # Calculate absolute times by adding trial start timestamp to relative times
                lick_start_absolute = None
                lick_end_absolute = None
                
                if trial_start_time_absolute is not None:
                    lick_start_absolute = trial_start_time_absolute + lick['lick_time']
                    if lick['lick_end_time'] is not None:
                        lick_end_absolute = trial_start_time_absolute + lick['lick_end_time']
                
                lick_event_entry = {
                    'subject_id': subject_id,
                    'session': session,
                    'trial': trial_num,
                    'lick_id': lick['lick_id'],
                    'port_number': lick['port_number'],
                    'lick_time': lick['lick_time'],  # Relative to trial start
                    'lick_duration': lick['lick_duration'],
                    'lick_start_absolute': lick_start_absolute,
                    'lick_end_absolute': lick_end_absolute,
                    'is_early': lick['is_early']
                }
                lick_events_data.append(lick_event_entry)
    
    # Create Port data from discovered ports
    for port_num, reward_size in port_reward_sizes.items():
        port_entry = {
            'subject_id': subject_id,
            'session': session,
            'port_number': port_num,
            'port_reward_size': reward_size
        }
        port_data.append(port_entry)
    
    # Create SessionTrial data based on number of trials
    session_trial_data = []
    if hasattr(session_data, 'RawEvents') and hasattr(session_data.RawEvents, 'Trial'):
        raw_trials = session_data.RawEvents.Trial
        if not isinstance(raw_trials, (list, np.ndarray)):
            raw_trials = [raw_trials]
        
        for trial_idx in range(len(raw_trials)):
            trial_num = trial_idx + 1
            start_time = session_data.TrialStartTimestamp[trial_idx] if hasattr(session_data, 'TrialStartTimestamp') and len(session_data.TrialStartTimestamp) > trial_idx else None
            session_trial_data.append({
                'subject_id': subject_id,
                'session': session,
                'trial': trial_num,
                'start_time': start_time
            })
    
    # Create DataFrames
    df_session_trial = pd.DataFrame(session_trial_data)
    df_port = pd.DataFrame(port_data)
    df_block = pd.DataFrame(block_data)
    df_reward = pd.DataFrame(reward_data)
    df_lick_events = pd.DataFrame(lick_events_data)
    
    print(f"\nCreated DataFrames:")
    print(f"SessionTrial: {len(df_session_trial)} rows")
    print(f"Port: {len(df_port)} rows")
    print(f"Block: {len(df_block)} rows")
    print(f"Reward: {len(df_reward)} rows")
    print(f"LickEvents: {len(df_lick_events)} rows")
    
    # Display sample data
    if not df_session_trial.empty:
        print("\nSessionTrial data:")
        print(df_session_trial.head())
    
    if not df_port.empty:
        print("\nPort data:")
        print(df_port)
    
    if not df_reward.empty:
        print("\nSample Reward data:")
        print(df_reward.head())
    
    if not df_lick_events.empty:
        print("\nSample LickEvents data:")
        print(df_lick_events.head())
    
    return df_session_trial, df_port, df_block, df_reward, df_lick_events

def save_dataframes_to_csv(df_session_trial, df_port, df_block, df_reward, df_lick_events, output_dir="./output", session_date=None):
    """Save DataFrames to CSV files in the specified directory.
    
    Args:
        df_session_trial: SessionTrial DataFrame
        df_port: Port DataFrame
        df_block: Block DataFrame
        df_reward: Reward DataFrame
        df_lick_events: LickEvents DataFrame
        output_dir: Base output directory
        session_date: Session date in YYYYMMDD format (extracted from filename)
    """
    
    # Create subfolder with session date if provided
    if session_date:
        output_dir = os.path.join(output_dir, session_date)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save SessionTrial DataFrame
        if not df_session_trial.empty:
            session_trial_path = os.path.join(output_dir, "session_trial.csv")
            df_session_trial.to_csv(session_trial_path, index=False)
            print(f"Saved SessionTrial data to: {session_trial_path}")
        
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
            f.write(f"SessionTrial records: {len(df_session_trial)}\n")
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

# COMMENTED OUT TO AVOID DJ CONNECTION - Uncomment when ready to insert to database
def insert_data_to_datajoint(df_session_trial, df_port, df_block, df_reward, df_lick_events):
    """Insert DataFrames into DataJoint tables.
    
    Insertion order:
    1. SessionTrial (depends on exp.Session)
    2. Port (depends on exp.Session)
    3. Block (depends on exp.Session and Port)
    4. Reward (depends on SessionTrial and Port)
    5. LickEvent (depends on SessionTrial and Port)
    """
    
    try:
        print("\nInserting data into DataJoint tables...")
        
        # Insert SessionTrial data first (depends on exp.Session)
        if not df_session_trial.empty:
            print(f"\nInserting {len(df_session_trial)} SessionTrial records...")
            print("Sample SessionTrial data:")
            print(df_session_trial.head())
            SessionTrial.insert(df_session_trial.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted SessionTrial records")
        
        # Insert Port data (depends on exp.Session)
        if not df_port.empty:
            print(f"\nInserting {len(df_port)} Port records...")
            print("Sample Port data:")
            print(df_port.head())
            Port.insert(df_port.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Port records")
        
        # Insert Block data (depends on exp.Session and Port)
        if not df_block.empty:
            print(f"\nInserting {len(df_block)} Block records...")
            print("Sample Block data:")
            print(df_block.head())
            Block.insert(df_block.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Block records")
        
        # Insert Reward data (depends on SessionTrial and Port)
        if not df_reward.empty:
            print(f"\nInserting {len(df_reward)} Reward records...")
            print("Sample Reward data:")
            print(df_reward.head(2))
            Reward.insert(df_reward.to_dict('records'), skip_duplicates=True)
            print(f"✓ Inserted Reward records")
        
        # Insert LickEvent data (depends on SessionTrial and Port)
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
        print("2. Verify column names match table definitions")
        print("3. Check foreign key constraints in table definitions")

if __name__ == "__main__":
    # Configuration
    subject_id = 105101  # Update this with your subject ID
    session = 1  # Update this with your session number
    
    # Path to the MATLAB file
    matlab_file = "NPC1_maze_pairs_fix_20250805_124140.mat"
    matlab_path = os.path.join(os.path.dirname(__file__), matlab_file)
    
    # Extract session date from filename (format: YYYYMMDD)
    # Filename pattern: *_YYYYMMDD_HHMMSS.mat
    import re
    date_match = re.search(r'_(\d{8})_', matlab_file)
    session_date = date_match.group(1) if date_match else None
    
    # Populate tables
    df_session_trial, df_port, df_block, df_reward, df_lick_events = populate_behavioral_tables(
        matlab_path, subject_id, session
    )
    
    if df_session_trial is not None:
        # Save DataFrames to CSV files
        output_directory = os.path.join(os.path.dirname(__file__), "output")
        save_dataframes_to_csv(df_session_trial, df_port, df_block, df_reward, df_lick_events, output_directory, session_date)
        
        # Uncomment the next line to actually insert data into DataJoint
        insert_data_to_datajoint(df_session_trial, df_port, df_block, df_reward, df_lick_events)
    else:
        print("Failed to process MATLAB data")
