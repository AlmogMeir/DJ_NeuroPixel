import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class UnitQualityType(dj.Lookup):
    definition = """
        #
        unit_quality: varchar(100)
        ---
        unit_quality_description: varchar(4000)
        """
    contents = [
        ['good', 'single unit'],
        ['ok', 'probably a single unit, but could be contaminated'],
        ['multi', 'multi unit'],
        ['all', 'all units'],
        ['ok or good', 'include both ok and good unit']
        ]