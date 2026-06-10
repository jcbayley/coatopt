import sys
import asyncio
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from coatopt_ui.app import analyze_stack, AnalyzeRequest, Layer, MaterialProp

def test_api_analyze_endpoint_direct():
    # Initial physical thicknesses from the prompt
    tphys = [
        123.756651, 217.102141, 122.323165, 214.986608, 121.736214, 214.513363, 
        121.474314, 214.857409, 121.558202, 215.174188, 121.848105, 214.361490, 
        121.467751, 214.157004, 121.509439, 214.345442, 121.107103, 213.806544, 
        120.635733, 213.625958, 120.579659, 213.453391, 120.052243, 212.675989, 
        119.865308, 212.276541, 119.639313, 212.153051, 119.221428, 211.582432, 
        119.424791, 211.581925, 119.263789, 211.261823, 118.904834, 210.961159, 
        118.850955, 210.725919, 118.670611, 211.257222, 118.840754, 212.337908, 
        121.091542, 291.088350
    ]

    materialLayer = [
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1
    ]

    layers = [Layer(thickness=t, material=m) for t, m in zip(tphys, materialLayer)]

    materialParams = {
        "0": MaterialProp(name="air", desc="Air", n=1.0, k=0.0),
        "1": MaterialProp(name="SiO2", desc="Silica - Thin film Room Temperature", n=1.45, k=0.0, a=0.0, alpha=5.1e-7, beta=0.000008, kappa=1.38, C=1641200, Y=70000000000, prat=0.19, phiM=0.000023),
        "2": MaterialProp(name="TiGermania", desc="Titania doped Germania - Room Temperature LMA", n=1.866, k=2e-7, a=1.0, alpha=0.000001282, beta=0.000024, kappa=33.0, C=2510000, Y=92000000000, prat=0.29, phiM=0.00009013672),
        "999": MaterialProp(name="air", desc="Air", n=1.0, k=0.0)
    }

    req = AnalyzeRequest(
        layers=layers,
        materialParams=materialParams,
        lambda_=1064.0,
        wBeam=0.062,
        Temp=293.0,
        polarisation="p",
        angle=0.0,
        target_lambdas=[1064.0, 532.0]
    )

    # Call endpoint function directly using asyncio.run
    response = asyncio.run(analyze_stack(req))
    
    # Verify response structure and status code
    import json
    assert response.status_code == 200
    
    data = json.loads(response.body.decode('utf-8'))
    
    # 1. Total Physical Thickness
    assert abs(data["total_thickness"] - 7420.11) < 0.1
    
    # 2. Optical Absorption
    assert abs(data["absorption_ppm"] - 0.35) < 0.05
    
    # 3. Brownian Noise (CTN) at 100 Hz
    assert abs(data["noise_100hz"]["brownian"] - 3.71268e-21) < 1e-23
    
    # 4. Transmission & Reflectivity at 1064 nm
    t_1064 = data["targets"]["1064.0"]["transmission"]
    r_1064 = data["targets"]["1064.0"]["reflectivity"]
    assert abs(t_1064 * 1e6 - 61.1872) < 0.01
    assert abs(r_1064 - 0.99994) < 1e-5
    
    # 5. Transmission at 532 nm
    t_532 = data["targets"]["532.0"]["transmission"]
    assert abs(t_532 * 100 - 2.51) < 0.05
    
    # 6. Material 1 stats (SiO2)
    m1 = data["materials"]["1"]
    assert m1["layers"] == 22
    assert abs(m1["thickness"] - 4768.29) < 0.1
    assert m1["n"] == 1.45
    
    # 7. Material 2 stats (TiGermania)
    m2 = data["materials"]["2"]
    assert m2["layers"] == 22
    assert abs(m2["thickness"] - 2651.82) < 0.1
    assert m2["n"] == 1.866
    
    print("\nAll UI endpoint direct call checks passed successfully!")

if __name__ == "__main__":
    test_api_analyze_endpoint_direct()
