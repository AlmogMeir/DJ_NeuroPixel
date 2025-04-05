import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class UnitPosition(dj.Part):
    definition = """
        # Estimated unit position in the brain
        -> EPHYS.Unit
        -> CF.CFAnnotationType
        ---
        -> LAB.Hemisphere
        -> LAB.BrainArea
        -> LAB.SkullReference
        unit_ml_location= null: decimal(8,3)    # um from ref; right is positive; based on manipulator coordinates (or histology) & probe config
        unit_ap_location= null: decimal(8,3)    # um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config
        unit_dv_location= null: decimal(8,3)    # um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config
        """