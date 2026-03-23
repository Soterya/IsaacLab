# Steps for Converting MuJoCo XMLs to IsaacSim USDs

- Use the below utility script to prime the xml for usd utility compatibility

```bash
# python scripts/korus/assets/xml/convert_xml_to_usd.py <input.xml> <output.xml>
python scripts/korus/utils/convert_xml1_to_xml2.py scripts/korus/assets/xml/0002/0002.xml scripts/korus/assets/xml/0002/humanoid_0002.xml
```

- Then convert it to usd format using the below isaaclab utility

```bash
# ./isaaclab.sh -p scripts/tools/convert_mjcf.py <input_xml.xml> <output_usd.usd>
./isaaclab.sh -p scripts/tools/convert_mjcf.py scripts/korus/assets/xml/0000/humanoid_0000.xml scripts/korus/assets/usd/0000/humanoid_0000.usd --import-sites 
```