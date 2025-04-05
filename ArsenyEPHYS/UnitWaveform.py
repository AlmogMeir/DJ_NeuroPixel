import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class UnitWaveform(dj.Part):
    definition = """
        # Estimated unit position in the brain
        -> EPHYS.Unit
        ---
        waveform: blob              # unit average waveform. time in samples, amplitude in microvolts.
        spk_width_ms: float         # unit average spike width, in ms
        sampling_fq: float          # Hz
        waveform_amplitude: float   # unit amplitude (peak) in microvolts
        """