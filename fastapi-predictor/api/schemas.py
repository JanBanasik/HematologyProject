from pydantic import BaseModel, Field

class BloodTest(BaseModel):
    RBC: float = Field(..., description="Red Blood Cells")
    HGB: float = Field(..., description="Hemoglobin")
    HCT: float = Field(..., description="Hematocrit")
    MCV: float = Field(..., description="Mean Corpuscular Volume")
    MCH: float = Field(..., description="Mean Corpuscular Hemoglobin")
    MCHC: float = Field(..., description="Mean Corpuscular Hemoglobin Concentration")
    RDW: float = Field(..., description="Red Cell Distribution Width")
    PLT: float = Field(..., description="Platelet Count")
    WBC: float = Field(..., description="White Blood Cells")
