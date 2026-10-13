from threading import Event

from ai_docs_agent.jobs import IndexJobs


def test_update_during_running_build_queues_followup():
    started, release = Event(), Event()

    class Engine:
        calls = 0

        def summary(self):
            return {"revision": self.calls}

        def build(self, *args):
            self.calls += 1
            if self.calls == 1:
                started.set()
                assert release.wait(3)
            return self.summary()

    engine = Engine()
    manager = IndexJobs(engine)
    first = manager.submit()
    assert started.wait(3)
    second = manager.submit()
    assert second["id"] != first["id"]
    release.set()
    manager.close()
    assert engine.calls == 2
    assert manager.get(second["id"])["status"] == "done"
    assert manager.public()["index"]["revision"] == 2
