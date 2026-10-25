"""Stage only registered historical data; no generation or model execution."""
from pathlib import Path
import hashlib,json,shutil

WORK=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def validate_references():
    settings=json.loads((WORK/'settings.json').read_text())
    sealed=json.loads((WORK/'references_manifest.json').read_text())
    for name,expected in sealed.items():
        assert digest(WORK/'references'/name)==expected, 'changed reference receipt '+name
    verified={}
    for tag,arms,key in [('exp030',['A','C25','D25'],'reference_work'),('exp031',['P25'],'reference_p25_work')]:
        parent=Path(settings[key])
        old=json.loads((WORK/'references'/tag/'settings.json').read_text())
        for name in ['training_seeds','start_stage','end_stage','train_bs','protection_bs_per_task','protection_tasks','data_sets','confirm_stages','preserved_data_sha256']:
            assert settings[name]==old[name], 'reference settings changed '+name
        receipt=json.loads((WORK/'references'/tag/'verification.json').read_text())
        assert receipt['passed'] and not receipt['failures']
        for arm in arms:
            for stage in settings['confirm_stages']:
                path=parent/'outputs/s5/confirm'/f'{arm}_{stage:011d}.npz'
                registered=receipt['verified_file_sha256'].get(str(path.resolve()))
                assert registered is not None and digest(path)==registered, 'reference rows not registered '+str(path)
                verified[str(path)]=registered
    out=WORK/'outputs';out.mkdir(exist_ok=True)
    (out/'reference_validation.json').write_text(json.dumps(dict(passed=True,verified_file_sha256=verified),indent=2)+'\n')
    return verified


def prepare():
    settings=json.loads((WORK/'settings.json').read_text())
    data=WORK/'outputs/data';data.mkdir(parents=True,exist_ok=True)
    for name,expected in settings['preserved_data_sha256'].items():
        source=Path(settings['reference_work'])/'outputs/data'/name
        assert digest(source)==expected, 'historical data changed '+name
        target=data/name
        if not target.exists():shutil.copyfile(source,target)
        assert digest(target)==expected
    files=validate_references()
    print(json.dumps(dict(prepared=True,preserved_hdf5=5,registered_reference_rows=len(files))))


if __name__=='__main__':prepare()
